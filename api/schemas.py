from pydantic import BaseModel
from typing import Dict, List, Optional

class BoundingBox(BaseModel):
    x1: int
    y1: int
    x2: int
    y2: int

class ShapeAnalysis(BaseModel):
    primary: str
    probabilities: Dict[str, float]

class Feature(BaseModel):
    label: str
    confidence: float
    boundingBoxCropped: Dict[str, int]

class Proportionality(BaseModel):
    harmonyScores: Dict[str, float]
    verticalThirds: Dict[str, str]
    horizontalFifths: Dict[str, str]
    rawMetrics: List[Dict]

class FaceAnalysis(BaseModel):
    confidence: float
    boundingBoxOriginal: BoundingBox
    shape: ShapeAnalysis
    features: List[Feature]
    proportionality: Proportionality

class AnalysisResult(BaseModel):
    face: FaceAnalysis

class VisualURL(BaseModel):
    annotated_image_url: str
    report_image_url: str

class AnalysisResponse(BaseModel):
    metadata: Dict
    sourceImage: Dict
    analysisResult: AnalysisResult
    visuals: VisualURL