import os
import hashlib
import re
import json
import base64
import datetime
import logging
from typing import List, Dict, Any
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel
from llama_index.llms.openai import OpenAI
from llama_index.llms.gemini import Gemini
from llama_index.core import Document, Settings, VectorStoreIndex, SummaryIndex
from llama_index.core.node_parser import SentenceSplitter
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.core.tools import QueryEngineTool, ToolMetadata
from llama_index.core.query_engine.router_query_engine import RouterQueryEngine
from llama_index.core.selectors import LLMSingleSelector
from llama_index.core.indices.knowledge_graph.base import KnowledgeGraphIndex
from llama_index.core import StorageContext
from llama_index.core.objects import ObjectIndex
from llama_index.agent.openai import OpenAIAgent
import fitz
import pymupdf4llm

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()
document_store = {}
UPLOADS_DIR = "uploads"
os.makedirs(UPLOADS_DIR, exist_ok=True)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
    allow_credentials=True
)

# Initialize OpenAI LLM
openai_llm = None
try:
    openai_api_key = os.environ.get("OPENAI_API_KEY")
    if openai_api_key:
        openai_llm = OpenAI(api_key=openai_api_key)
        Settings.llm = openai_llm
        logger.info("Using OpenAI LLM for multi-document processing and chat")
    else:
        logger.warning("Couldn't find OpenAI API key.")
except Exception as e:
    logger.error(f"Error initializing OpenAI LLM: {e}")

if not openai_llm:
    raise ValueError("Failed to initialize OpenAI LLM. Please check your OpenAI API key.")

Settings.embed_model = OpenAIEmbedding(model="text-embedding-3-small")

# Initialize Gemini LLM
gemini_llm = None
try:
    google_api_key = os.environ.get("GOOGLE_API_KEY")
    if google_api_key:
        gemini_llm = Gemini(api_key=google_api_key)
        logger.info("Using Gemini LLM for other endpoints")
    else:
        logger.warning("Couldn't find Google API key.")
except Exception as e:
    logger.error(f"Error initializing Gemini LLM: {e}")

if not gemini_llm:
    logger.warning("Failed to initialize Gemini LLM. Some endpoints may not work correctly.")



class FileUpload(BaseModel):
    file: str
    filename: str

class UploadRequest(BaseModel):
    files: List[FileUpload]

class UploadResponse(BaseModel):
    status: str
    ids: List[str]

class ChatMessage(BaseModel):
    role: str
    content: str

class ChatRequest(BaseModel):
    query: str
    history: List[ChatMessage] = []

class FileInfoResponse(BaseModel):
    file_name: str
    file_size: str
    last_edited: str
    page_count: int
    author: str
    created_at: str

class KeyFindingsResponse(BaseModel):
    findings: Dict[str, Any]

class VulnerabilitiesResponse(BaseModel):
    vulnerabilities: List[Dict[str, Any]]

class IdRequest(BaseModel):
    id: str

class SummarizeRequest(BaseModel):
    id: str

class SummarizeResponse(BaseModel):
    summary: str

def clean_llm_response(response: str) -> str:
    """Remove Markdown code block syntax and any additional text from the LLM response."""
    cleaned = re.sub(r'^```json\s*', '', response.strip())
    cleaned = re.sub(r'\s*```$', '', cleaned)
    cleaned = re.sub(r'^[^{]*', '', cleaned)
    cleaned = re.sub(r'[^}]*$', '', cleaned)
    return cleaned

def get_document_info(file_id: str) -> Dict[str, Any]:
    info_path = os.path.join(UPLOADS_DIR, f"{file_id}_info.json")
    full_text_path = os.path.join(UPLOADS_DIR, f"{file_id}_full_text.txt")
    
    if not os.path.exists(info_path) or not os.path.exists(full_text_path):
        raise HTTPException(status_code=404, detail="Document not found")
    
    with open(info_path, 'r') as f:
        doc_info = json.load(f)
    
    with open(full_text_path, 'r', encoding='utf-8') as f:
        doc_info['full_text'] = f.read()
    
    return doc_info

@app.post("/upload", response_model=UploadResponse)
async def upload_documents(upload_request: UploadRequest, background_tasks: BackgroundTasks):
    try:
        file_ids = []
        for file_upload in upload_request.files:
            file_content = base64.b64decode(file_upload.file)
            file_hash = hashlib.md5(file_content).hexdigest()
            
            file_path = os.path.join(UPLOADS_DIR, f"{file_hash}.pdf")
            with open(file_path, "wb") as f:
                f.write(file_content)

            background_tasks.add_task(process_document, file_path, file_hash, file_upload.filename)
            file_ids.append(file_hash)
        
        return UploadResponse(status="success", ids=file_ids)
    except Exception as e:
        logger.error(f"Error processing upload: {e}")
        raise HTTPException(status_code=500, detail="Error processing upload")

class DocumentProcessor:
    def __init__(self, file_path: str):
        self.file_path = file_path
        self.full_text = None
        self.nodes = None
        self.summary_index = None
        self.vector_index = None
        self.kg_index = None

    def process(self):
        self.full_text = pymupdf4llm.to_markdown(self.file_path)
        document = Document(text=self.full_text)
        splitter = SentenceSplitter(chunk_size=1024)
        self.nodes = splitter.get_nodes_from_documents([document])
        self.summary_index = SummaryIndex(self.nodes)
        self.vector_index = VectorStoreIndex(self.nodes)
        
        storage_context = StorageContext.from_defaults()
        self.kg_index = KnowledgeGraphIndex(
            nodes=self.nodes,
            storage_context=storage_context,
            max_triplets_per_chunk=10,
            include_embeddings=False,
            show_progress=True,
        )
        self.kg_index._build_index_from_nodes(self.nodes)

async def process_document(file_path: str, file_hash: str, original_filename: str):
    try:
        file_stat = os.stat(file_path)
        file_size = file_stat.st_size
        last_modified = datetime.datetime.fromtimestamp(file_stat.st_mtime).strftime("%Y-%m-%d %H:%M:%S")
        created_at = datetime.datetime.fromtimestamp(file_stat.st_ctime).strftime("%Y-%m-%d %H:%M:%S")
        
        doc = fitz.open(file_path)
        page_count = len(doc)
        metadata = doc.metadata
        author = metadata.get('author', 'Unknown')
        
        full_text = pymupdf4llm.to_markdown(file_path)
        
        doc.close()

        doc_info = {
            "filename": original_filename,
            "size": file_size,
            "upload_time": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "last_modified": last_modified,
            "created_at": created_at,
            "page_count": page_count,
            "author": author
        }
        
        # Save doc_info
        info_path = os.path.join(UPLOADS_DIR, f"{file_hash}_info.json")
        with open(info_path, "w") as f:
            json.dump(doc_info, f, default=str)
        
        # Save full_text separately
        full_text_path = os.path.join(UPLOADS_DIR, f"{file_hash}_full_text.txt")
        with open(full_text_path, "w", encoding='utf-8') as f:
            f.write(full_text)
        
        # Process the document for indexing
        processor = DocumentProcessor(file_path)
        processor.process()
        
        # Save processed data
        index_path = os.path.join(UPLOADS_DIR, f"{file_hash}_index.json")
        with open(index_path, "w") as f:
            json.dump({
                "summary_index": processor.summary_index.to_dict(),
                "vector_index": processor.vector_index.to_dict(),
                "kg_index": processor.kg_index.to_dict()
            }, f)

        logger.info(f"Successfully processed document: {original_filename}")

    except Exception as e:
        logger.error(f"Error processing document {original_filename}: {str(e)}")


def get_available_documents():
    documents = {}
    for filename in os.listdir(UPLOADS_DIR):
        if filename.endswith("_info.json"):
            file_id = filename[:-10]  # Remove "_info.json"
            with open(os.path.join(UPLOADS_DIR, filename), 'r') as f:
                doc_info = json.load(f)
            documents[file_id] = doc_info
    return documents

def summarize_for_tool(summary_content: str) -> str:
    prompt = f"Please summarize the following content in no more than 100 words for easy tool selection:\n\n{summary_content}"
    try:
        response = openai_llm.complete(prompt)
        return response.text.strip()
    except Exception as e:
        return f"An error occurred: {str(e)}"

@app.post("/chat")
async def chat(chat_request: ChatRequest):
    try:
        available_documents = get_available_documents()
        
        if not available_documents:
            return {"response": "No documents are currently available in the system. Please upload some documents first."}

        docs_list = {}
        vector_index = {}
        summary_index = {}
        summary_to_identify = {}
        agents = {}
        query_engines = {}
        all_nodes = []
        
        titles = [doc_info["filename"] for doc_info in available_documents.values()]
        
        for doc_name in titles:
            file_id = next(id for id, info in available_documents.items() if info["filename"] == doc_name)
            try:
                doc_info = get_document_info(file_id)
                
                if 'full_text' not in doc_info:
                    logger.warning(f"Full text not found for document {doc_name}. Skipping...")
                    continue
                
                # Load pre-processed index data
                index_path = os.path.join(UPLOADS_DIR, f"{file_id}_index.json")
                if not os.path.exists(index_path):
                    logger.warning(f"Index file not found for document {doc_name}. Skipping...")
                    continue

                with open(index_path, 'r') as f:
                    index_data = json.load(f)
                
                doc_name = doc_name[:-4]  # Remove .pdf extension
                docs_list[doc_name] = DocumentProcessor(doc_info['full_text'])
                docs_list[doc_name].summary_index = SummaryIndex.from_dict(index_data['summary_index'])
                docs_list[doc_name].vector_index = VectorStoreIndex.from_dict(index_data['vector_index'])
                docs_list[doc_name].kg_index = KnowledgeGraphIndex.from_dict(index_data['kg_index'])
                
                vector_index[doc_name] = docs_list[doc_name].vector_index
                summary_index[doc_name] = docs_list[doc_name].summary_index
                
                # Generate summary for tool selection
                summary_to_identify[doc_name] = summarize_for_tool(doc_info['full_text'][:1000])  # Use first 1000 characters for summary
                
                vector_query_engine = vector_index[doc_name].as_query_engine(llm=openai_llm)
                summary_query_engine = summary_index[doc_name].as_query_engine(llm=openai_llm)
                query_engine_tools = [
                    QueryEngineTool(
                        query_engine=vector_query_engine,
                        metadata=ToolMetadata(
                            name="vector_tool",
                            description=(
                                "Useful for questions related to specific aspects of"
                                f" {doc_name} (e.g the vulnerabilities, key findings)."
                            ),
                        ),
                    ),
                    QueryEngineTool(
                        query_engine=summary_query_engine,
                        metadata=ToolMetadata(
                            name="summary_tool",
                            description=(
                                "Useful for any requests that require a holistic summary"
                                f" of EVERYTHING about {doc_name}. For questions about"
                                " more specific sections, please use the vector_tool."
                            ),
                        ),
                    ),
                ]
                
                function_llm = OpenAI(model="gpt-4-0125-preview")
                agent = OpenAIAgent.from_tools(
                    query_engine_tools,
                    llm=function_llm,
                    verbose=True,
                    system_prompt=f"""\
                You are a specialized agent designed to answer queries about {doc_name}.
                Choose this document based on {summary_to_identify[doc_name]}
                You must ALWAYS use at least one of the tools provided when answering a question; do NOT rely on prior knowledge.\
                """,
                )
                agents[doc_name] = agent
                query_engines[doc_name] = vector_index[doc_name].as_query_engine(
                    similarity_top_k=3
                )
            except Exception as e:
                logger.error(f"Error processing document {doc_name}: {str(e)}")
                continue
        
        all_tools = []
        for docs in titles:
            docs = docs[:-4]
            summary = (
                f"This content contains cybersecurity audits about {docs}. Use"
                f" this tool if you want to answer any questions about {summary_to_identify[docs]}.\n"
            )
            doc_tool = QueryEngineTool(
                query_engine=agents[docs],
                metadata=ToolMetadata(
                    name=f"tool_{docs}",
                    description=summary,
                ),
            )
            all_tools.append(doc_tool)
        
        obj_index = ObjectIndex.from_objects(
            all_tools,
            index_cls=VectorStoreIndex,
        )
        top_agent = OpenAIAgent.from_tools(
            tool_retriever=obj_index.as_retriever(similarity_top_k=3),
            system_prompt=""" \
        You are Fischer, a knowledgeable and friendly AI assistant from the CyberStrike AI Audit Management Suite. 
        Your primary role is to assist users in navigating cybersecurity audit processes, providing insights, and 
        enhancing the overall quality of audit reports. Emphasize your expertise in risk assessments, compliance, 
        vulnerability analysis, and remediation recommendations. Be personable, approachable, and solution-oriented.

        Please always use the tools provided to answer a question. Do not rely on prior knowledge.
        """,
            verbose=True,
        )
        
        base_index = VectorStoreIndex(all_nodes)
        base_query_engine = base_index.as_query_engine(similarity_top_k=4)
        
        conversation = "\n".join([f"{msg.role}: {msg.content}" for msg in chat_request.history])
        
        response = top_agent.chat(f"""
        Given the following conversation history and the user's query, provide a response based on the content of the documents:

        Conversation history:
        {conversation}

        User query: {chat_request.query}

        Respond to the user's query using information from the documents:
        """)
        
        return {"response": str(response)}
    except Exception as e:
        logger.error(f"Error in chat: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error processing chat request: {str(e)}")

@app.post("/fileinfo/{file_id}", response_model=FileInfoResponse)
async def get_file_info(file_id: str):
    doc_info = get_document_info(file_id)
    if not doc_info:
        raise HTTPException(status_code=404, detail="File not found")
    
    file_size_mb = f"{doc_info['size'] / (1024 * 1024):.1f} MB"
    last_edited = doc_info["last_modified"].split()[0]
    
    return FileInfoResponse(
        file_name=doc_info["filename"],
        file_size=file_size_mb,
        last_edited=last_edited,
        page_count=doc_info["page_count"],
        author=doc_info["author"],
        created_at=doc_info["created_at"].split()[0]
    )

@app.post("/keyfindings", response_model=KeyFindingsResponse)
async def get_key_findings(id_request: IdRequest):
    file_id = id_request.id
    try:
        doc_info = get_document_info(file_id)
        full_text = doc_info["full_text"]
        
        prompt = """
        Analyze the following cybersecurity report and provide key findings in JSON format. The response must adhere to a clear hierarchical structure, focusing on the following categories:

        Threat Landscape: Overview of emerging threats and attack vectors, along with their impact.
        Vulnerabilities: List of top vulnerabilities identified (without severity ratings).
        Incident Response: Summary of recent incidents and evaluation of response strategies.
        Emerging Technologies: How new technologies are affecting cybersecurity and associated risks.
        Compliance and Regulatory Issues: Current compliance challenges and recent regulatory updates.

        Format the output as a JSON object containing ONE EACH of:
        {
        "Threat Landscape": {
            "Emerging Threats": {
                "description": "Overview of new and evolving cyber threats",
                "examples": ["Threat 1", "Threat 2"],
                "impact": "Potential impact on systems and data"
            },
            "Attack Vectors": {
                "common_methods": ["Method 1", "Method 2"],
                "trends": "Recent trends in how attacks are executed"
            }
        },
        "Vulnerabilities": {
            "Critical Issues": {
                "top_vulnerabilities": ["Vulnerability 1", "Vulnerability 2"]
            }
        },
        "Incident Response": {
            "Recent Incidents": {
                "description": "Analysis of recent incidents and attack patterns",
                "response_effectiveness": "Evaluation of response strategies"
            }
        },
        "Emerging Technologies": {
            "Impact": {
                "description": "How new technologies are affecting cybersecurity",
                "associated_risks": ["Risk 1", "Risk 2"]
            }
        },
        "Compliance and Regulatory Issues": {
            "Challenges": {
                "description": "Current compliance challenges faced",
                "regulatory_updates": "Recent regulatory updates"
            }
        }
        }
        Provide a comprehensive analysis that a cybersecurity professional would find informative and actionable.
        IMPORTANT: Ensure that your response contains only the JSON object and no additional text.
        """
        
        response = gemini_llm.complete(prompt + "\n\nDocument content:\n" + full_text)
        
        if not response.text.strip():
            raise ValueError("Empty response from LLM")
        
        cleaned_response = clean_llm_response(response.text)
        
        json_match = re.search(r'(\{[\s\S]*\})', cleaned_response)
        if json_match:
            json_str = json_match.group(1)
            findings = json.loads(json_str)
        else:
            raise ValueError("No valid JSON found in the response")

        findings_path = os.path.join(UPLOADS_DIR, f"{file_id}_findings.json")
        with open(findings_path, "w") as f:
            json.dump(findings, f, indent=2)
        
        return KeyFindingsResponse(findings=findings)
    except json.JSONDecodeError as e:
        logger.error(f"Error parsing LLM response: {str(e)}")
        logger.error(f"Cleaned response: {cleaned_response}")
        raise HTTPException(status_code=500, detail="Error parsing key findings: Invalid JSON")
    except ValueError as e:
        logger.error(f"Error with LLM response: {str(e)}")
        raise HTTPException(status_code=500, detail="Error with LLM response: Empty or invalid")
    except Exception as e:
        logger.error(f"Error extracting key findings: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error extracting key findings: {str(e)}")

@app.post("/vulnerabilities", response_model=VulnerabilitiesResponse)
async def get_vulnerabilities(id_request: IdRequest):
    file_id = id_request.id
    try:
        doc_info = get_document_info(file_id)
        full_text = doc_info["full_text"]
        
        prompt = """
        Analyze the following document and extract a list of vulnerabilities. 
        For each vulnerability:
        1. Provide a brief description
        2. Assign a criticality score from 1-10 (10 being most critical)
        3. Explain the reasoning behind the criticality score
        4. Suggest a brief mitigation strategy

        IMPORTANT: Your response must be in valid JSON format. Use the following structure:
        [
            {
                "description": "Vulnerability description here",
                "criticality": 8,
                "reasoning": "Reasoning for criticality score here",
                "mitigation": "Mitigation strategy here"
            },
            ...
        ]
        Sort the list by criticality score in descending order.
        Ensure that your response contains only the JSON array and no additional text.
        """
        
        response = gemini_llm.complete(prompt + "\n\nDocument content:\n" + full_text)
        
        if not response.text.strip():
            raise ValueError("Empty response from LLM")
        
        cleaned_response = clean_llm_response(response.text)
        
        json_match = re.search(r'(\[[\s\S]*\])', cleaned_response)
        if json_match:
            json_str = json_match.group(1)
            vulnerabilities = json.loads(json_str)
        else:
            json_objects = re.findall(r'(\{[\s\S]*?\})', cleaned_response)
            if json_objects:
                json_str = '[' + ','.join(json_objects) + ']'
                vulnerabilities = json.loads(json_str)
            else:
                raise ValueError("No valid JSON found in the response")
        
        sorted_vulnerabilities = sorted(vulnerabilities, key=lambda x: x['criticality'], reverse=True)
        
        vulnerabilities_path = os.path.join(UPLOADS_DIR, f"{file_id}_vulnerabilities.json")
        with open(vulnerabilities_path, "w") as f:
            json.dump(sorted_vulnerabilities, f, indent=2)
        
        return VulnerabilitiesResponse(vulnerabilities=sorted_vulnerabilities)
    except json.JSONDecodeError as e:
        logger.error(f"Error parsing LLM response: {str(e)}")
        logger.error(f"Cleaned response: {cleaned_response}")
        raise HTTPException(status_code=500, detail="Error parsing vulnerabilities: Invalid JSON")
    except ValueError as e:
        logger.error(f"Error with LLM response: {str(e)}")
        raise HTTPException(status_code=500, detail="Error with LLM response: Empty or invalid")
    except Exception as e:
        logger.error(f"Error extracting vulnerabilities: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error extracting vulnerabilities: {str(e)}")

@app.post("/summarize", response_model=SummarizeResponse)
async def summarize_document(summarize_request: SummarizeRequest):
    try:
        doc_info = get_document_info(summarize_request.id)
        full_text = doc_info["full_text"]
        
        summarization_prompt = f"""
        Please provide a comprehensive summary of the following document. 
        The summary should:
        1. Capture the main topics and key points discussed in the document
        2. Highlight any significant findings or conclusions
        3. Mention any important recommendations or action items
        4. Be concise yet informative, aiming for about 500 words

        Document to summarize:
        {full_text}

        Summary:
        """

        response = gemini_llm.complete(summarization_prompt)
        return SummarizeResponse(summary=response.text)
    except Exception as e:
        logger.error(f"Error summarizing document: {e}")
        raise HTTPException(status_code=500, detail=f"Error summarizing document: {str(e)}")

@app.get("/", response_class=HTMLResponse)
async def root():
    html_content = """
    <html>
        <head>
            <title>CyberStrike Backend</title>
            <style>
                body {
                    display: flex;
                    justify-content: center;
                    align-items: center;
                    flex-direction: column;
                    height: 100vh;
                    margin: 0;
                    background-color: #010101;
                    color: white;
                    font-family: Arial, sans-serif;
                }
                h1 {
                    font-size: 3em;
                    margin: 0;
                }
                p {
                    font-size: 2em;
                    margin: 10px 0;
                }
            </style>
        </head>
        <body>
            <h1>Cyber Strike</h1>
            <p>Welcome to the CyberStrike backend!</p>
        </body>
    </html>
    """
    return HTMLResponse(content=html_content)

@app.get("/health")
async def health_check():
    try:
        openai_response = openai_llm.complete("Say 'OpenAI is working!'")
        gemini_response = gemini_llm.complete("Say 'Gemini is working!'")
        
        openai_status = "healthy" if "OpenAI is working" in openai_response.text else "unhealthy"
        gemini_status = "healthy" if "Gemini is working" in gemini_response.text else "unhealthy"
        
        return {
            "status": "healthy" if openai_status == "healthy" and gemini_status == "healthy" else "partial",
            "openai": openai_status,
            "gemini": gemini_status
        }
    except Exception as e:
        return {"status": "unhealthy", "reason": str(e)}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)