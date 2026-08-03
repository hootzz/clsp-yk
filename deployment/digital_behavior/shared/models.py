"""
HQS Data Stream - 공통 데이터 모델
proposal의 Data Stream 구조에 맞게 정의
"""
from dataclasses import dataclass, field, asdict
from datetime import datetime
from typing import Optional
import json


@dataclass
class DeviceInteractionEvent:
    """
    Data Stream의 Device Interaction Behavior 단위 이벤트
    
    proposal 구조:
      Time Stamp | User State Component | Sub-Component
      Device Interaction Behavior | device_type | Usage log | Summarize(LLM)
    """
    timestamp: str                        # ISO 8601
    device_type: str                      # "pc" | "android"
    
    # Sub-components
    app: str                              # 앱/프로세스 이름
    title: Optional[str] = None          # 창 제목 또는 화면 이름
    url: Optional[str] = None            # 브라우저 URL (PC only)
    duration_seconds: float = 0.0        # 해당 앱에서 머문 시간
    
    # Android-specific
    event_type: Optional[str] = None     # "app_switch" | "click" | "text_input" | "screen_change"
    element: Optional[str] = None        # 클릭한 UI 요소 (Android only)
    
    # Screenshot path (변경 시점에 캡처)
    screenshot_path: Optional[str] = None
    
    # 세션 관리
    session_id: Optional[str] = None
    
    # LLM 요약 (3단계에서 채워짐)
    llm_summary: Optional[str] = None
    
    # 원본 raw data (디버깅용)
    raw: Optional[dict] = field(default=None, repr=False)

    def to_dict(self) -> dict:
        return asdict(self)

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False, indent=2)


@dataclass
class Session:
    """
    비활동 간격으로 분리된 작업 세션
    여러 DeviceInteractionEvent를 묶어서 LLM 요약의 단위가 됨
    """
    session_id: str
    device_type: str
    start_time: str
    end_time: Optional[str] = None
    events: list = field(default_factory=list)
    llm_summary: Optional[str] = None

    @property
    def duration_seconds(self) -> float:
        if not self.end_time:
            return 0.0
        start = datetime.fromisoformat(self.start_time)
        end = datetime.fromisoformat(self.end_time)
        return (end - start).total_seconds()