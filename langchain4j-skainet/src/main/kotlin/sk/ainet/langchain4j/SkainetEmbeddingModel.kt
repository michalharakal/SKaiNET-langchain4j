package sk.ainet.langchain4j

import dev.langchain4j.data.embedding.Embedding
import dev.langchain4j.data.segment.TextSegment
import dev.langchain4j.model.embedding.EmbeddingModel
import dev.langchain4j.model.output.Response
import dev.langchain4j.model.output.TokenUsage
import kotlinx.coroutines.runBlocking
import sk.ainet.apps.bert.BertIngestion
import sk.ainet.apps.bert.BertModelConfig
import sk.ainet.apps.bert.BertRuntime
import sk.ainet.apps.bert.BertRuntimeWeights
import sk.ainet.apps.bert.HuggingFaceTokenizer
import sk.ainet.context.DirectCpuExecutionContext
import sk.ainet.io.JvmRandomAccessSource
import sk.ainet.io.safetensors.SafeTensorsParametersLoader
import sk.ainet.lang.tensor.Tensor
import sk.ainet.lang.tensor.data.FloatArrayTensorData
import sk.ainet.lang.types.DType
import sk.ainet.lang.types.FP32
import java.nio.file.Path
import kotlin.io.path.exists
import kotlin.io.path.readText

/**
 * LangChain4j [EmbeddingModel] backed by SKaiNET's BERT runtime.
 *
 * Loads a HuggingFace-compatible BERT model from a local directory containing
 * `model.safetensors`, `vocab.txt`, and `config.json`.
 *
 * Usage:
 * ```kotlin
 * val model = SkainetEmbeddingModel.builder()
 *     .modelDir(Path.of("models/e5-small-v2"))
 *     .build()
 *
 * val embedding = model.embed("Hello world").content()
 * ```
 */
class SkainetEmbeddingModel private constructor(
    private val bertRuntime: BertRuntime<FP32>,
    private val tokenizer: HuggingFaceTokenizer,
    private val name: String
) : EmbeddingModel {

    override fun embedAll(textSegments: List<TextSegment>): Response<List<Embedding>> {
        var totalTokens = 0
        val embeddings = textSegments.map { segment ->
            val tokOutput = tokenizer.encodeWithMetadata(segment.text())
            totalTokens += tokOutput.inputIds.size
            val tensor = bertRuntime.encode(
                tokOutput.inputIds,
                tokOutput.attentionMask,
                tokOutput.tokenTypeIds
            )
            val floats = tensor.toFloatArray()
            Embedding.from(floats)
        }
        return Response.from(embeddings, TokenUsage(totalTokens))
    }

    companion object {
        @JvmStatic
        fun builder(): Builder = Builder()

        private fun <T : DType> Tensor<T, Float>.toFloatArray(): FloatArray {
            val data = this.data
            if (data is FloatArrayTensorData<*>) return data.buffer.copyOf()
            return data.copyToFloatArray()
        }
    }

    class Builder {
        private var modelDir: Path? = null
        private var modelName: String = "skainet-bert"

        fun modelDir(path: Path) = apply { this.modelDir = path }
        fun modelName(name: String) = apply { this.modelName = name }

        fun build(): SkainetEmbeddingModel {
            val dir = requireNotNull(modelDir) { "modelDir must be set" }
            require(dir.toFile().exists()) { "Model directory not found: $dir" }

            // Parse config.json
            val configPath = dir.resolve("config.json")
            require(configPath.exists()) { "config.json not found in $dir" }
            val config = parseConfigJson(configPath.readText(), dir)

            println("BERT config: hidden=${config.hiddenSize}, layers=${config.numHiddenLayers}, heads=${config.numAttentionHeads}")

            // Load tokenizer
            val vocabPath = dir.resolve("vocab.txt")
            require(vocabPath.exists()) { "vocab.txt not found in $dir" }
            val tokenizer = HuggingFaceTokenizer.fromVocabTxt(vocabPath.readText())
            println("Tokenizer loaded (vocab=${tokenizer.vocabSize})")

            // Load model weights
            val ctx = DirectCpuExecutionContext()
            val ingestion = BertIngestion<FP32>(ctx, FP32::class, config)

            // Find safetensors files
            val mainSafetensors = resolveModelFile(dir)
            val loaders = mutableListOf(
                SafeTensorsParametersLoader(
                    sourceProvider = { JvmRandomAccessSource.open(mainSafetensors.toString()) }
                )
            )

            // Check for sentence-transformers 2_Dense projection layer
            val denseSafetensors = dir.resolve("2_Dense/model.safetensors")
            if (denseSafetensors.exists()) {
                loaders.add(
                    SafeTensorsParametersLoader(
                        sourceProvider = { JvmRandomAccessSource.open(denseSafetensors.toString()) }
                    )
                )
            }

            println("Loading BERT weights ...")
            val weights: BertRuntimeWeights<FP32> = runBlocking {
                if (loaders.size == 1) {
                    ingestion.load(loaders[0])
                } else {
                    // Use multi-loader path for projection layer
                    sk.ainet.apps.bert.loadBertWeights(loaders, ctx, FP32::class, config)
                }
            }

            val runtime = BertRuntime(ctx, weights, FP32::class)
            println("SKaiNET BERT model ready.")

            return SkainetEmbeddingModel(runtime, tokenizer, modelName)
        }

        private fun resolveModelFile(modelDir: Path): Path {
            for (name in listOf("model.safetensors", "pytorch_model.safetensors")) {
                val p = modelDir.resolve(name)
                if (p.exists()) return p
            }
            val found = modelDir.toFile().listFiles()?.firstOrNull { it.extension == "safetensors" }
            if (found != null) return found.toPath()
            error("No .safetensors file found in $modelDir")
        }

        private fun parseConfigJson(json: String, modelDir: Path): BertModelConfig {
            fun extractInt(source: String, key: String, default: Int): Int {
                val pattern = Regex("\"$key\"\\s*:\\s*(\\d+)")
                return pattern.find(source)?.groupValues?.get(1)?.toIntOrNull() ?: default
            }
            fun extractDouble(source: String, key: String, default: Double): Double {
                val pattern = Regex("\"$key\"\\s*:\\s*([\\d.eE\\-+]+)")
                return pattern.find(source)?.groupValues?.get(1)?.toDoubleOrNull() ?: default
            }

            val denseConfigPath = modelDir.resolve("2_Dense/config.json")
            val projDim = if (denseConfigPath.exists()) {
                val denseJson = denseConfigPath.readText()
                extractInt(denseJson, "out_features", 0).let { if (it > 0) it else null }
            } else {
                null
            }

            return BertModelConfig(
                vocabSize = extractInt(json, "vocab_size", 30522),
                hiddenSize = extractInt(json, "hidden_size", 384),
                numHiddenLayers = extractInt(json, "num_hidden_layers", 6),
                numAttentionHeads = extractInt(json, "num_attention_heads", 12),
                intermediateSize = extractInt(json, "intermediate_size", 1536),
                maxPositionEmbeddings = extractInt(json, "max_position_embeddings", 512),
                typeVocabSize = extractInt(json, "type_vocab_size", 2),
                layerNormEps = extractDouble(json, "layer_norm_eps", 1e-12),
                projectionDim = projDim
            )
        }
    }
}
