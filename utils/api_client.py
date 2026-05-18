import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
import logging
from typing import Dict, Any, Optional
from decouple import config

# Setup logger
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

class BoQIntelligenceAPI:
    """
    Enterprise API Client for connecting Streamlit to the FastAPI Backend.
    Handles PropelAuth token injection, automatic retries, and file streaming.
    """
    
    def __init__(self, base_url: str = config("API_BASE_URL", default="http://localhost:8000")):
        self.base_url = base_url.rstrip("/")
        self.session = self._build_retry_session()

    def _build_retry_session(self) -> requests.Session:
        """Configures a session with robust connection retries for large payloads."""
        session = requests.Session()
        retries = Retry(
            total=3,
            backoff_factor=1,
            status_forcelist=[500, 502, 503, 504]
        )
        session.mount("http://", HTTPAdapter(max_retries=retries))
        session.mount("https://", HTTPAdapter(max_retries=retries))
        return session

    def _get_headers(self, access_token: Optional[str] = None) -> Dict[str, str]:
        """Injects the PropelAuth Bearer token into the headers."""
        headers = {"Accept": "application/json"}
        if access_token:
            headers["Authorization"] = f"Bearer {access_token}"
        return headers

    def post_json(
        self,
        endpoint: str,
        payload: Dict[str, Any],
        access_token: str,
        timeout: int = 30,
    ) -> Dict[str, Any]:
        """Send JSON data to the backend."""
        url = f"{self.base_url}/{endpoint.lstrip('/')}"
        try:
            response = self.session.post(
                url,
                json=payload,
                headers=self._get_headers(access_token),
                timeout=timeout,
            )
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            logger.error(f"API POST Error at {endpoint}: {str(e)}")
            raise e


    def predict_batch(
        self,
        items: list[dict],
        access_token: str,
        timeout: int = 120,
    ) -> dict:
        """
        Send a batch of BoQ item dicts to the inference engine.

        Args:
            items:        List of raw BoQ item dicts (any schema — the API normalises them).
            access_token: PropelAuth Bearer token.
            timeout:      Request timeout in seconds (default 120 for large batches).

        Returns:
            Parsed JSON response dict from the API.
        """
        return self.post_json(
            "/api/v1/predict",
            payload={"items": items},
            access_token=access_token,
            timeout=timeout,
        )
