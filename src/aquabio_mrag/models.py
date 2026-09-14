from __future__ import annotations

from typing import Any, Literal, TypedDict

from pydantic import BaseModel, ConfigDict, Field


Modality = Literal[
    "text",
    "image_caption",
    "image_text_pair",
    "species_card",
    "pdf_text",
    "pdf_figure",
]


class RAGDocument(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str
    source_type: str
    species_id: str = ""
    modality: Modality
    content: str
    embedding_text: str
    metadata: dict[str, Any] = Field(default_factory=dict)


class ImageDocument(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str
    source_type: Literal["image_doc"] = "image_doc"
    species_id: str
    english_name: str
    chinese_name: str
    image_path: str
    image_url: str
    source_page: str
    license: str
    license_url: str = ""
    author: str = ""
    caption: str
    visual_keywords: list[str] = Field(default_factory=list)
    embedding_text: str
    width: int = 0
    height: int = 0
    bing_discovery_url: str = ""


class MultimodalPair(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str
    source_type: Literal["multimodal_pair"] = "multimodal_pair"
    species_id: str
    image_id: str
    text_ids: list[str]
    image_caption: str
    rag_context: str
    embedding_text: str


class PDFRegistryEntry(BaseModel):
    model_config = ConfigDict(extra="forbid")

    doc_id: str
    title: str
    source_org: str
    doc_type: Literal[
        "species_identification_guide",
        "ecology_monitoring_manual",
        "education_material",
        "conservation_report",
        "research_paper",
    ]
    language: str = "English"
    local_path: str
    source_url: str
    usage: list[str]
    priority: int = 1


class RouteDecision(BaseModel):
    model_config = ConfigDict(extra="forbid")

    task_type: Literal[
        "text_qa",
        "followup_text_qa",
        "image_qa",
        "multimodal_qa",
        "comparison_qa",
        "source_trace",
        "pdf_qa",
        "detection_qa",
    ]
    need_vlm: bool
    need_text_retrieval: bool
    need_image_retrieval: bool
    need_multimodal_retrieval: bool
    need_pdf_retrieval: bool
    need_detection: bool


class EvaluationResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    passed: bool
    score: float = Field(ge=0.0, le=1.0)
    reason: str
    retry_target: Literal[
        "none", "rewrite", "retrieval", "vision", "answer"
    ] = "none"


class AquaBioState(TypedDict, total=False):
    session_id: str
    session_initialized: bool
    conversation_history: list[dict[str, Any]]
    memory_summary: dict[str, Any]
    pending_memory_turn: dict[str, Any]
    resolved_query: str
    resolved_species_ids: list[str]
    followup_detected: bool
    need_clarification: bool
    clarification_question: str
    response_mode: str
    response_constraints: list[str]
    session_file: str
    original_query: str
    rewritten_query: str
    image_path: str | None
    image_caption: str | None
    vision_failed: bool
    visual_features: list[str]
    candidate_species: list[str]
    detected_species_ids: list[str]
    route: dict[str, Any]
    selected_tools: list[str]
    tool_plan: list[str]
    tool_observations: list[str]
    tool_calls: list[dict[str, Any]]
    react_step: int
    hitl_enabled: bool
    human_review: dict[str, Any]
    graph_context: list[dict[str, Any]]
    fused_context: list[dict[str, Any]]
    graph_trace: list[str]
    text_context: list[dict[str, Any]]
    web_context: list[dict[str, Any]]
    image_context: list[dict[str, Any]]
    multimodal_context: list[dict[str, Any]]
    pdf_context: list[dict[str, Any]]
    final_context: str
    draft_answer: str
    generation_failed: bool
    provider_fallback: bool
    evaluation_result: dict[str, Any]
    final_answer: str
    retry_count: int
    trace: list[str]
    warnings: list[str]
    runtime_options: dict[str, bool]
    extra_context: str
    web_research_plan: dict[str, Any]
    detection_result: dict[str, Any]
    detection_comparison: dict[str, Any]
    detection_visualization_path: str
