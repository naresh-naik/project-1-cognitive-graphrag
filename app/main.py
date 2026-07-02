import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from app.config import settings
from app.pipeline import CognitiveGraphRAGPipeline

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)

logger = logging.getLogger("app.main")

pipeline = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global pipeline

    logger.info("Starting Cognitive GraphRAG pipeline...")

    try:
        pipeline = CognitiveGraphRAGPipeline()
        logger.info("Cognitive GraphRAG pipeline started.")
        yield
    finally:
        if pipeline:
            pipeline.close()
            logger.info("Cognitive GraphRAG pipeline closed.")


app = FastAPI(
    title=settings.PROJECT_NAME,
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class IndexRequest(BaseModel):
    title: str = Field(..., example="Quantum Research Lab")
    content: str = Field(..., example="Dr. Aris Thorne leads the Quantum Materials Division.")
    source: str = Field(default="manual")


class QueryRequest(BaseModel):
    query: str = Field(..., example="Who leads the Quantum Materials Division?")


@app.get("/")
def root():
    return {
        "message": "Cognitive GraphRAG API is running",
        "project": settings.PROJECT_NAME,
    }


@app.get("/health")
def health():
    return {
        "status": "healthy",
        "project": settings.PROJECT_NAME,
    }


@app.post("/index", status_code=status.HTTP_201_CREATED)
def index_document(request: IndexRequest):
    if pipeline is None:
        raise HTTPException(status_code=503, detail="Pipeline is not ready.")

    try:
        return pipeline.index_document(
            title=request.title,
            content=request.content,
            source=request.source,
        )
    except Exception as error:
        logger.exception("Indexing failed: %s", error)
        raise HTTPException(status_code=500, detail=str(error))


@app.post("/query")
def query_document(request: QueryRequest):
    if pipeline is None:
        raise HTTPException(status_code=503, detail="Pipeline is not ready.")

    try:
        return pipeline.query(request.query)
    except Exception as error:
        logger.exception("Query failed: %s", error)
        raise HTTPException(status_code=500, detail=str(error))
