from pydantic import BaseModel, Field
from typing import List, Optional
from enum import Enum
from datetime import datetime

class UnitType(str, Enum):
    concept = "concept"
    definition = "definition"
    algorithm = "algorithm"
    procedure = "procedure"
    example = "example"
    comparison = "comparison"
    fact = "fact"
    reference = "reference"

class RelationshipType(str, Enum):
    part_of = "part_of"
    related_to = "related_to"
    depends_on = "depends_on"
    contrasts_with = "contrasts_with"

class Relationship(BaseModel):
    type: RelationshipType
    target_id: str

class KnowledgeUnit(BaseModel):
    id: str = Field(..., description="Unique identifier for the knowledge unit")
    title: str = Field(..., description="Title or concise summary of the knowledge unit")
    type: UnitType
    document: str = Field(..., description="Source document name")
    section: Optional[str] = Field(None, description="Section or chapter name")
    page_start: int = Field(..., description="Starting page number")
    page_end: int = Field(..., description="Ending page number")
    keywords: List[str] = Field(default_factory=list, description="List of keywords")
    content: str = Field(..., description="The semantic content itself (Markdown format)")
    relationships: List[Relationship] = Field(default_factory=list, description="Relationships to other units")

class Manifest(BaseModel):
    source_pdf: str
    kb_name: str
    generator: str = "RAG Knowledge Base Generator MVP"
    version: str = "1.0.0"
    num_units: int
    format: str = "Markdown + JSON"
    has_metadata: bool = True
    has_relationships: bool = True


class JobStatus(str, Enum):
    pending = "pending"
    processing = "processing"
    done = "done"
    failed = "failed"


class JobInfo(BaseModel):
    job_id: str
    status: JobStatus = JobStatus.pending
    filename: str
    num_units: Optional[int] = None
    zip_path: Optional[str] = None
    error: Optional[str] = None
    logs: List[str] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
