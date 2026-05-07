from typing import Any

from dify_plugin import ToolProvider
from dify_plugin.errors.tool import ToolProviderCredentialValidationError

from zep_cloud.client import Zep
from zep_cloud.core.api_error import ApiError


class ZepProvider(ToolProvider):
    def _validate_credentials(self, credentials: dict[str, Any]) -> None:
        try:
            api_key = self.runtime.credentials["zep_api_key"]
            base_url = self.runtime.credentials["zep_base_url"]
            client = Zep(api_key=api_key, base_url=base_url)
            client.memory.get(session_id="test")
        except Exception as e:
            if isinstance(e, ApiError) and e.status_code == 401:
                raise ToolProviderCredentialValidationError(str(e)) from e
