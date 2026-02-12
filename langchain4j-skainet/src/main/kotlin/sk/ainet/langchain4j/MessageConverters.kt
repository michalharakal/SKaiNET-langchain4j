package sk.ainet.langchain4j

import dev.langchain4j.agent.tool.ToolSpecification
import dev.langchain4j.data.message.AiMessage
import dev.langchain4j.data.message.ChatMessage
import dev.langchain4j.data.message.SystemMessage
import dev.langchain4j.data.message.ToolExecutionResultMessage
import dev.langchain4j.data.message.UserMessage
import dev.langchain4j.model.chat.request.json.JsonObjectSchema
import dev.langchain4j.model.chat.request.json.JsonSchemaElement
import dev.langchain4j.model.chat.request.json.JsonStringSchema
import dev.langchain4j.model.chat.request.json.JsonIntegerSchema
import dev.langchain4j.model.chat.request.json.JsonNumberSchema
import dev.langchain4j.model.chat.request.json.JsonBooleanSchema
import dev.langchain4j.model.chat.request.json.JsonArraySchema
import dev.langchain4j.model.output.FinishReason
import dev.langchain4j.model.output.TokenUsage
import kotlinx.serialization.json.JsonObject
import kotlinx.serialization.json.JsonPrimitive
import kotlinx.serialization.json.buildJsonObject
import kotlinx.serialization.json.put
import kotlinx.serialization.json.putJsonArray
import kotlinx.serialization.json.putJsonObject
import sk.ainet.apps.kllama.chat.ChatRole
import sk.ainet.apps.kllama.chat.ToolCall
import sk.ainet.apps.kllama.chat.ToolDefinition
import sk.ainet.apps.kllama.chat.ChatMessage as SkainetChatMessage

/**
 * Convert a LangChain4j [ChatMessage] to a SKaiNET [SkainetChatMessage].
 */
fun ChatMessage.toSkainet(): SkainetChatMessage = when (this) {
    is SystemMessage -> SkainetChatMessage(
        role = ChatRole.SYSTEM,
        content = text()
    )
    is UserMessage -> SkainetChatMessage(
        role = ChatRole.USER,
        content = singleText()
    )
    is AiMessage -> SkainetChatMessage(
        role = ChatRole.ASSISTANT,
        content = text() ?: "",
        toolCalls = if (hasToolExecutionRequests()) {
            toolExecutionRequests().map { req ->
                ToolCall(
                    id = req.id() ?: "call_0",
                    name = req.name(),
                    arguments = kotlinx.serialization.json.Json.parseToJsonElement(req.arguments()).let {
                        it as? JsonObject ?: JsonObject(emptyMap())
                    }
                )
            }
        } else null
    )
    is ToolExecutionResultMessage -> SkainetChatMessage(
        role = ChatRole.TOOL,
        content = text(),
        toolCallId = id()
    )
    else -> SkainetChatMessage(
        role = ChatRole.USER,
        content = toString()
    )
}

/**
 * Convert a list of LangChain4j [ChatMessage]s to SKaiNET format.
 */
fun List<ChatMessage>.toSkainet(): List<SkainetChatMessage> = map { it.toSkainet() }

/**
 * Convert a LangChain4j [ToolSpecification] to a SKaiNET [ToolDefinition].
 */
fun ToolSpecification.toSkainet(): ToolDefinition {
    val parametersJson = if (parameters() != null) {
        jsonSchemaToJsonObject(parameters())
    } else {
        JsonObject(emptyMap())
    }
    return ToolDefinition(
        name = name(),
        description = description() ?: "",
        parameters = parametersJson
    )
}

/**
 * Convert a list of LangChain4j [ToolSpecification]s to SKaiNET [ToolDefinition]s.
 */
fun List<ToolSpecification>.toSkainetToolDefs(): List<ToolDefinition> = map { it.toSkainet() }

/**
 * Convert a SKaiNET [ToolCall] to a LangChain4j [dev.langchain4j.agent.tool.ToolExecutionRequest].
 */
fun ToolCall.toLangchain4j(): dev.langchain4j.agent.tool.ToolExecutionRequest =
    dev.langchain4j.agent.tool.ToolExecutionRequest.builder()
        .id(id)
        .name(name)
        .arguments(arguments.toString())
        .build()

/**
 * Convert a [JsonSchemaElement] to a kotlinx.serialization [JsonObject].
 */
private fun jsonSchemaToJsonObject(schema: JsonSchemaElement): JsonObject = buildJsonObject {
    when (schema) {
        is JsonObjectSchema -> {
            put("type", "object")
            if (schema.description() != null) put("description", schema.description())
            if (schema.properties() != null && schema.properties().isNotEmpty()) {
                putJsonObject("properties") {
                    schema.properties().forEach { (name, propSchema) ->
                        put(name, jsonSchemaToJsonObject(propSchema))
                    }
                }
            }
            if (schema.required() != null && schema.required().isNotEmpty()) {
                putJsonArray("required") {
                    schema.required().forEach { add(JsonPrimitive(it)) }
                }
            }
        }
        is JsonStringSchema -> {
            put("type", "string")
            if (schema.description() != null) put("description", schema.description())
        }
        is JsonIntegerSchema -> {
            put("type", "integer")
            if (schema.description() != null) put("description", schema.description())
        }
        is JsonNumberSchema -> {
            put("type", "number")
            if (schema.description() != null) put("description", schema.description())
        }
        is JsonBooleanSchema -> {
            put("type", "boolean")
            if (schema.description() != null) put("description", schema.description())
        }
        is JsonArraySchema -> {
            put("type", "array")
            if (schema.description() != null) put("description", schema.description())
        }
        else -> {
            put("type", "string")
        }
    }
}
