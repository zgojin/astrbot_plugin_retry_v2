import asyncio

import astrbot.api.message_components as Comp
from astrbot.api import AstrBotConfig, logger
from astrbot.api.event import AstrMessageEvent, filter
from astrbot.api.provider import LLMResponse, ProviderRequest
from astrbot.api.star import Context, Star, register

# 正常完成的标志
NORMAL_FINISH_REASONS = {"stop", "tool_calls"}


@register(
    "astrbot_plugin_retry_v2",
    "长安某",
    "一个基于事件钩子的、处理空回复和截断等非完整响应的重试插件",
    "1.0.0",
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

        default_keywords = [
            "API 返回的内容为空",
        ]
        self.retry_keywords = self.config.get("retry_keywords", default_keywords)
        logger.info("astrbot_plugin_retry_v2已加载")

    def _get_request_key(self, event: AstrMessageEvent) -> str:
        """获取用于追踪请求的唯一ID"""
        try:
            if event.message_obj and event.message_obj.message_id:
                return str(event.message_obj.message_id)
        except AttributeError:
            pass
        return str(id(event))

    @filter.on_llm_request()
    async def store_llm_request(self, event: AstrMessageEvent, req: ProviderRequest):
        """备份原始请求"""
        key = self._get_request_key(event)
        self.pending_requests[key] = req
        logger.debug(f"已成功备份请求 (Key: {key})")

    @filter.after_message_sent(priority=-100)
    async def cleanup_after_sent(self, event: AstrMessageEvent):
        """当消息成功发送后，清理对应的请求备份，防止内存泄漏"""
        key = self._get_request_key(event)
        if key in self.pending_requests:
            self.pending_requests.pop(key)
            logger.debug(f"请求已成功处理并发送，清理备份 (Key: {key})")

    def _is_response_failed(self, resp: LLMResponse) -> tuple[bool, str]:
        """检查LLM响应是否失败"""
        if not resp or not resp.completion_text:
            return True, "返回内容为空"

        try:
            # 尝试获取finish_reason
            finish_reason = resp.raw_completion.choices[0].finish_reason
        except (AttributeError, IndexError):
            # 捕获异常，说明finish_reason字段缺失，视为失败
            return True, "finish_reason缺失"

        # 如果字段存在，但值为null(None)，也视为失败
        if finish_reason is None:
            return True, "finish_reason为空"

        # 原有逻辑：检查finish_reason的值是否在正常列表中
        if finish_reason.lower() not in NORMAL_FINISH_REASONS:
            return True, f"finish_reason异常({finish_reason})"

        return False, ""

    async def _perform_retry(self, original_req: ProviderRequest) -> LLMResponse:
        """执行一次LLM请求"""
        return await self.context.get_using_provider().text_chat(
            prompt=original_req.prompt,
            contexts=original_req.contexts,
            image_urls=original_req.image_urls,
            func_tool=original_req.func_tool,
            system_prompt=original_req.system_prompt,
        )

    async def _execute_retry_loop(
        self,
        event: AstrMessageEvent,
        original_req: ProviderRequest,
        initial_reason: str,
    ) -> tuple[bool, LLMResponse | None]:
        """重试循环逻辑"""
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

            logger.info(
                f"对请求 {key} 进行第 {retry_count}/{self.max_retries} 次重试..."
            )

            try:
                new_response = await self._perform_retry(original_req)
                is_retry_failed, retry_reason = self._is_response_failed(new_response)

                if not is_retry_failed:
                    logger.info(f"重试成功 (Key: {key})")
                    return True, new_response
                else:
                    logger.warning(
                        f"第 {retry_count} 次重试仍失败 (Key: {key}, 原因: {retry_reason})"
                    )
            except Exception as e:
                logger.error(f"第 {retry_count} 次重试时发生异常 (Key: {key}): {e}")
                if retry_count >= self.max_retries:
                    logger.error(
                        f"已达到最大重试次数，且最后一次尝试时发生异常 {key}，终止"
                    )
                    return False, None

        logger.error(f"请求 {key} 已达到最大重试次数，放弃")
        return False, None

    @filter.on_decorating_result(priority=100)
    async def retry_on_exception_failure(self, event: AstrMessageEvent):
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

        # 调用封装的重试逻辑
        initial_reason = f"触发关键词: '{trigger_keyword}'"
        is_success, new_response = await self._execute_retry_loop(
            event, original_req, initial_reason
        )

        if is_success and new_response:
            event.set_result(event.plain_result(new_response.completion_text))
        else:
            event.set_result(
                event.plain_result(self.fallback_reply.format(reason=initial_reason))
            )

    @filter.on_llm_response(priority=-10)
    async def retry_on_llm_failure(self, event: AstrMessageEvent, resp: LLMResponse):
        key = self._get_request_key(event)
        original_req = self.pending_requests.get(key)

        if not original_req:
            return

        is_failed, reason = self._is_response_failed(resp)
        if not is_failed:
            return

        is_success, new_response = await self._execute_retry_loop(
            event, original_req, reason
        )

        if is_success and new_response:
            # 重试成功，更新旧的响应对象
            resp.role = new_response.role
            resp.completion_text = new_response.completion_text
            resp.raw_completion = new_response.raw_completion
            resp.tools_call_args = new_response.tools_call_args
            resp.tools_call_name = new_response.tools_call_name
            resp.result_chain = new_response.result_chain
        else:
            # 重试最终失败，发送fallback消息并阻止后续处理
            await event.send(
                event.plain_result(self.fallback_reply.format(reason=reason))
            )
            # 通过修改resp内容为空让后续处理知道已经失败
            resp.completion_text = ""
            resp.raw_completion = None
