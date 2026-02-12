package sk.ainet.langchain4j

import dev.langchain4j.data.message.AiMessage
import dev.langchain4j.model.chat.ChatModel
import dev.langchain4j.model.chat.request.ChatRequest
import dev.langchain4j.model.chat.response.ChatResponse
import dev.langchain4j.model.output.FinishReason
import dev.langchain4j.model.output.TokenUsage
import kotlinx.coroutines.runBlocking
import sk.ainet.apps.kllama.CpuAttentionBackend
import sk.ainet.apps.kllama.GGUFTokenizer
import sk.ainet.apps.kllama.LlamaIngestion
import sk.ainet.apps.kllama.LlamaLoadConfig
import sk.ainet.apps.kllama.LlamaRuntime
import sk.ainet.apps.kllama.agent.GenerateResult
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
 * LangChain4j [ChatModel] backed by SKaiNET's KLlama runtime.
 *
 * Loads a GGUF model from a local file and runs inference entirely on the CPU
 * using SKaiNET's native Kotlin inference engine.
 *
 * Usage:
 * ```kotlin
 * val model = SkainetChatModel.builder()
 *     .modelPath(Path.of("models/TinyLlama-1.1B-Chat-v1.0.Q4_K_M.gguf"))
 *     .temperature(0.3f)
 *     .build()
 *
 * val response = model.chat(
 *     SystemMessage.from("You are a helpful assistant."),
 *     UserMessage.from("Hello!")
 * )
 * ```
 */
class SkainetChatModel private constructor(
    private val runtime: LlamaRuntime<FP32>,
    private val tokenizer: GGUFTokenizer,
    private val chatTemplate: ChatTemplate,
    private val temperature: Float,
    private val maxTokens: Int
) : ChatModel {

    override fun doChat(chatRequest: ChatRequest): ChatResponse {
        val skainetMessages = chatRequest.messages().toSkainet()

        val toolDefs = chatRequest.toolSpecifications()
            ?.toSkainetToolDefs()
            ?: emptyList()

        val prompt = chatTemplate.apply(skainetMessages, toolDefs, addGenerationPrompt = true)
        val promptTokens = tokenizer.encode(prompt)

        val result: GenerateResult
        synchronized(runtime) {
            runtime.reset()
            result = runtime.generateUntilStop(
                prompt = promptTokens,
                maxTokens = maxTokens,
                eosTokenId = tokenizer.eosId,
                temperature = temperature,
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

        return ChatResponse.builder()
            .aiMessage(aiMessage)
            .tokenUsage(TokenUsage(promptTokens.size, result.tokens.size))
            .finishReason(finishReason)
            .build()
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

        fun build(): SkainetChatModel {
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

            println("SKaiNET model ready.")
            return SkainetChatModel(runtime, tokenizer, chatTemplate, temperature, maxTokens)
        }
    }
}
