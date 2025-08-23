import asyncio
import json

import astrbot.api.message_components as Comp
from astrbot.api import AstrBotConfig, logger
from astrbot.api.event import AstrMessageEvent, filter
from astrbot.api.provider import LLMResponse, ProviderRequest
from astrbot.api.star import Context, Star, register

try:
    from openai.types.chat.chat_completion import ChatCompletion
except ImportError:
    ChatCompletion = None

try:
    from google.genai.types import GenerateContentResponse
except ImportError:
    GenerateContentResponse = None


# 定义正常完成的标志
NORMAL_FINISH_REASONS = {"stop", "tool_calls"}


@register(
    "astrbot_plugin_retry_v2",
    "长安某",
    "一个基于事件钩子的、处理空回复和截断等非完整响应的重试插件",
    "1.0.1",
    "https://github.com/zgojin/astrbot_plugin_retry_v2",
)
class FinalLLMRetryPlugin(Star):
    def __init__(self, context: Context, config: AstrBotConfig):
        super().__init__(context)
        self.config = config
        self.pending_requests: dict[str, ProviderRequest] = {}
        self.max_retries = self.config.get("max_retries", 3)
        self.retry_delay = self.config.get("retry_delay", 2)
        self.use_exponential_backoff = self.config.get("retry_delay_mode", True)
        self.fallback_reply = self.config.get(
            "fallback_reply", "抱歉，请求多次失败({reason})，请稍后再试"
        )
        self.retry_keywords = self.config.get("retry_keywords", ["API 返回的内容为空"])
        self.log_each_response = self.config.get("log_each_response", True)
        logger.info("astrbot_plugin_retry_v2 已加载")

    def _get_request_key(self, event: AstrMessageEvent) -> str:
        try:
            if event.message_obj and event.message_obj.message_id:
                return str(event.message_obj.message_id)
        except AttributeError:
            pass
        return str(id(event))

    @filter.on_llm_request()
    async def store_llm_request(self, event: AstrMessageEvent, req: ProviderRequest):
        key = self._get_request_key(event)
        self.pending_requests[key] = req
        logger.debug(f"已成功备份请求 (Key: {key})")

    @filter.after_message_sent(priority=-100)
    async def cleanup_after_sent(self, event: AstrMessageEvent):
        key = self._get_request_key(event)
        if key in self.pending_requests:
            self.pending_requests.pop(key)
            logger.debug(f"请求已成功处理并发送，清理备份 (Key: {key})")

    def _is_response_failed(self, resp: LLMResponse) -> tuple[bool, str]:
        """检查 LLM 响应"""
        raw_data = resp.raw_completion

        if raw_data is None:
            return True, "原始响应(raw_completion)为空"

        finish_reason_str = None
        has_text_content = False
        is_a_valid_tool_call = False

        # 优先检查 resp.completion_text
        if resp.completion_text and resp.completion_text.strip():
            has_text_content = True

        # OpenAI 格式
        if ChatCompletion and isinstance(raw_data, ChatCompletion):
            if raw_data.choices:
                choice = raw_data.choices[0]
                finish_reason_str = getattr(choice, "finish_reason", None)
                if getattr(choice.message, "tool_calls", None):
                    is_a_valid_tool_call = True
                if (
                    choice.message
                    and choice.message.content
                    and choice.message.content.strip()
                ):
                    has_text_content = True

        # Gemini 格式
        elif GenerateContentResponse and isinstance(raw_data, GenerateContentResponse):
            if raw_data.candidates:
                candidate = raw_data.candidates[0]
                gemini_finish_reason = getattr(candidate, "finish_reason", None)
                if gemini_finish_reason:
                    finish_reason_str = gemini_finish_reason.name.lower()

                if (
                    hasattr(candidate, "content")
                    and hasattr(candidate.content, "parts")
                    and candidate.content.parts
                ):
                    for part in candidate.content.parts:
                        if hasattr(part, "function_call") and getattr(
                            part, "function_call", None
                        ):
                            is_a_valid_tool_call = True
                        if (
                            hasattr(part, "text")
                            and getattr(part, "text", "")
                            and part.text.strip()
                        ):
                            has_text_content = True

        # 如果是工具调用，不判定为失败，交由框架处理。
        if is_a_valid_tool_call:
            return False, ""

        # 如果完成原因异常且不是工具调用
        if finish_reason_str and finish_reason_str not in NORMAL_FINISH_REASONS:
            return True, f"完成原因异常({finish_reason_str})"

        # 没有文本内容，不是工具调用
        if not has_text_content:
            return True, "响应内容为空"

        # 其他情况视为成功
        return False, ""

    async def _perform_retry(self, original_req: ProviderRequest) -> LLMResponse:
        return await self.context.get_using_provider().text_chat(
            prompt=original_req.prompt,
            contexts=original_req.contexts,
            image_urls=original_req.image_urls,
            func_tool=original_req.func_tool,
            system_prompt=original_req.system_prompt,
        )

    async def _log_response(self, resp: LLMResponse, key: str):
        if not self.log_each_response:
            return
        logger.info("----------捕获到 LLM 响应 ----------")
        logger.info(
            f"LLMResponse (metadata): { {k: v for k, v in resp.__dict__.items() if k != 'raw_completion'} }"
        )

        raw_data = resp.raw_completion
        if raw_data is None:
            logger.warning("本次响应的 raw_completion 为 None")
        else:
            try:
                if hasattr(raw_data, "model_dump_json"):
                    dumped_json = raw_data.model_dump_json(indent=2)
                    logger.info(
                        f"检测到 Pydantic 模型 ({type(raw_data).__name__})，其内容如下:\n{dumped_json}"
                    )
                elif isinstance(raw_data, (dict, list)):
                    pretty_json_str = json.dumps(raw_data, indent=2, ensure_ascii=False)
                    logger.info(f"检测到字典/列表，其内容如下:\n{pretty_json_str}")
                else:
                    logger.info(
                        f"raw_completion 类型: {type(raw_data)}, 内容如下:\n{str(raw_data)}"
                    )
            except Exception as e:
                logger.error(f"打印 raw_completion 时出错: {e}", exc_info=True)
                logger.info(f"尝试强行打印 raw_completion:\n{str(raw_data)}")
        logger.info("---------- 响应打印结束 ----------")

    async def _execute_retry_loop(
        self,
        event: AstrMessageEvent,
        original_req: ProviderRequest,
        initial_reason: str,
    ) -> tuple[bool, LLMResponse | None]:
        key = self._get_request_key(event)
        logger.warning(
            f"检测到可重试的错误 (原因: '{initial_reason}'), Key: {key}准备重试(最多 {self.max_retries} 次)..."
        )

        delay = self.retry_delay
        for i in range(self.max_retries):
            retry_count = i + 1
            if i > 0:
                await asyncio.sleep(delay)
                if self.use_exponential_backoff:
                    delay = min(delay * 2, 30)

            logger.info(f"请求 {key} 第 {retry_count}/{self.max_retries} 次重试...")

            try:
                new_response = await self._perform_retry(original_req)
                await self._log_response(new_response, key)
                is_retry_failed, retry_reason = self._is_response_failed(new_response)

                if not is_retry_failed:
                    logger.info(f"重试成功 (Key: {key})")
                    return True, new_response
                else:
                    logger.warning(
                        f"第 {retry_count} 次重试失败 (Key: {key}, 原因: {retry_reason})"
                    )
            except Exception as e:
                logger.error(
                    f"第 {retry_count} 次重试时发生异常 (Key: {key}): {e}",
                    exc_info=True,
                )

        logger.error(f"所有 {self.max_retries} 次重试均失败 (Key: {key})")
        return False, None

    @filter.on_decorating_result(priority=100)
    async def retry_on_exception_failure(self, event: AstrMessageEvent):
        """基于关键词触发重试"""
        key = self._get_request_key(event)
        original_req = self.pending_requests.get(key)
        if not original_req:
            return

        result = event.get_result()
        if not result or not result.chain:
            return

        text_parts = [
            comp.text
            for comp in result.chain
            if isinstance(comp, Comp.Plain) and hasattr(comp, "text")
        ]
        result_text = "".join(text_parts)

        trigger_keyword = ""
        result_text_lower = result_text.lower()
        for keyword in self.retry_keywords:
            if keyword.lower() in result_text_lower:
                trigger_keyword = keyword
                break

        if not trigger_keyword:
            return

        initial_reason = f"触发关键词: '{trigger_keyword}'"
        is_success, new_response = await self._execute_retry_loop(
            event, original_req, initial_reason
        )

        if is_success and new_response:
            event.set_result(new_response.result_chain)
        else:
            event.set_result(
                event.plain_result(self.fallback_reply.format(reason=initial_reason))
            )

    @filter.on_llm_response(priority=10)
    async def retry_on_llm_failure(self, event: AstrMessageEvent, resp: LLMResponse):
        """基于响应触发重试"""
        key = self._get_request_key(event)
        original_req = self.pending_requests.get(key)
        if not original_req:
            return

        await self._log_response(resp, key)
        is_failed, reason = self._is_response_failed(resp)
        if not is_failed:
            return

        is_success, new_response = await self._execute_retry_loop(
            event, original_req, reason
        )

        if is_success and new_response:
            resp.__dict__.update(new_response.__dict__)
        else:
            await event.send(
                event.plain_result(self.fallback_reply.format(reason=reason))
            )
            event.stop_event()
