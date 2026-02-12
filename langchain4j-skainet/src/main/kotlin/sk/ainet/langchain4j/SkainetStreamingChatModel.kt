package sk.ainet.langchain4j

import dev.langchain4j.data.message.AiMessage
import dev.langchain4j.model.chat.StreamingChatModel
import dev.langchain4j.model.chat.request.ChatRequest
import dev.langchain4j.model.chat.response.ChatResponse
import dev.langchain4j.model.chat.response.StreamingChatResponseHandler
import dev.langchain4j.model.output.FinishReason
import dev.langchain4j.model.output.TokenUsage
import kotlinx.coroutines.runBlocking
import sk.ainet.apps.kllama.CpuAttentionBackend
import sk.ainet.apps.kllama.GGUFTokenizer
import sk.ainet.apps.kllama.LlamaIngestion
import sk.ainet.apps.kllama.LlamaLoadConfig
import sk.ainet.apps.kllama.LlamaRuntime
import sk.ainet.apps.kllama.agent.generateUntilStop
import sk.ainet.apps.kllama.chat.ChatMLTemplate
import sk.ainet.apps.kllama.chat.ChatTemplate
import sk.ainet.apps.kllama.chat.Llama3ChatTemplate
import sk.ainet.apps.kllama.chat.ToolCallParser
import sk.ainet.context.DirectCpuExecutionContext
import sk.ainet.io.JvmRandomAccessSource
import sk.ainet.io.gguf.llama.LlamaWeightLoader
import sk.ainet.lang.types.FP32
import java.nio.file.Path

/**
 * LangChain4j [StreamingChatModel] backed by SKaiNET's KLlama runtime.
 *
 * Streams tokens one at a time via the [StreamingChatResponseHandler] callbacks.
 */
class SkainetStreamingChatModel private constructor(
    private val runtime: LlamaRuntime<FP32>,
    private val tokenizer: GGUFTokenizer,
    private val chatTemplate: ChatTemplate,
    private val temperature: Float,
    private val maxTokens: Int
) : StreamingChatModel {

    override fun doChat(chatRequest: ChatRequest, handler: StreamingChatResponseHandler) {
        Thread.startVirtualThread {
            try {
                val skainetMessages = chatRequest.messages().toSkainet()

                val toolDefs = chatRequest.toolSpecifications()
                    ?.toSkainetToolDefs()
                    ?: emptyList()

                val prompt = chatTemplate.apply(skainetMessages, toolDefs, addGenerationPrompt = true)
                val promptTokens = tokenizer.encode(prompt)

                val result = synchronized(runtime) {
                    runtime.reset()
                    runtime.generateUntilStop(
                        prompt = promptTokens,
                        maxTokens = maxTokens,
                        eosTokenId = tokenizer.eosId,
                        temperature = temperature,
                        onToken = { tokenId ->
                            val tokenText = tokenizer.decode(tokenId)
                            handler.onPartialResponse(tokenText)
                        },
                        decode = { tokenizer.decode(it) }
                    )
                }

                val text = result.text
                val toolCalls = ToolCallParser.parse(text)

                val aiMessage = if (toolCalls.isNotEmpty()) {
                    AiMessage.from(toolCalls.map { it.toLangchain4j() })
                } else {
                    AiMessage.from(text)
                }

                val finishReason = when {
                    toolCalls.isNotEmpty() -> FinishReason.TOOL_EXECUTION
                    result.stoppedByEos -> FinishReason.STOP
                    else -> FinishReason.LENGTH
                }

                val chatResponse = ChatResponse.builder()
                    .aiMessage(aiMessage)
                    .tokenUsage(TokenUsage(promptTokens.size, result.tokens.size))
                    .finishReason(finishReason)
                    .build()

                handler.onCompleteResponse(chatResponse)
            } catch (e: Throwable) {
                handler.onError(e)
            }
        }
    }

    companion object {
        @JvmStatic
        fun builder(): Builder = Builder()
    }

    class Builder {
        private var modelPath: Path? = null
        private var temperature: Float = 0.7f
        private var maxTokens: Int = 512
        private var templateName: String = "llama3"

        fun modelPath(path: Path) = apply { this.modelPath = path }
        fun temperature(temperature: Float) = apply { this.temperature = temperature }
        fun maxTokens(maxTokens: Int) = apply { this.maxTokens = maxTokens }
        fun templateName(name: String) = apply { this.templateName = name }

        fun build(): SkainetStreamingChatModel {
            val path = requireNotNull(modelPath) { "modelPath must be set" }
            require(path.toFile().exists()) { "Model file not found: $path" }

            val ctx = DirectCpuExecutionContext()
            val ingestion = LlamaIngestion<FP32>(
                ctx = ctx,
                dtype = FP32::class,
                config = LlamaLoadConfig(
                    quantPolicy = LlamaWeightLoader.QuantPolicy.DEQUANTIZE_TO_FP32,
                    allowQuantized = false
                )
            )

            println("Loading GGUF model from $path ...")
            val runtimeWeights = runBlocking {
                ingestion.loadStreaming { JvmRandomAccessSource.open(path.toString()) }
            }

            val backend = CpuAttentionBackend<FP32>(ctx, runtimeWeights, FP32::class)
            val runtime = LlamaRuntime<FP32>(ctx, runtimeWeights, backend, FP32::class)

            println("Loading tokenizer ...")
            val tokenizer = JvmRandomAccessSource.open(path.toString()).use { source ->
                GGUFTokenizer.fromRandomAccessSource(source)
            }

            val chatTemplate: ChatTemplate = when (templateName.lowercase()) {
                "chatml" -> ChatMLTemplate()
                else -> Llama3ChatTemplate()
            }

            println("SKaiNET streaming model ready.")
            return SkainetStreamingChatModel(runtime, tokenizer, chatTemplate, temperature, maxTokens)
        }
    }
}
