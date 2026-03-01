plugins {
    alias(libs.plugins.kotlin.jvm)
    alias(libs.plugins.kotlin.serialization)
}

kotlin {
    jvmToolchain(21)
}

dependencies {
    // LangChain4j core API
    api(libs.langchain4j.core)

    // SKaiNET modules
    implementation(libs.skainet.kllama)
    implementation(libs.skainet.kllama.agent)
    implementation(libs.skainet.bert)
    implementation(libs.skainet.llm)
    implementation(libs.skainet.backend.cpu)
    implementation(libs.skainet.lang.core)
    implementation(libs.skainet.compile.core)
    implementation(libs.skainet.io.core)
    implementation(libs.skainet.io.gguf)
    implementation(libs.skainet.io.safetensors)

    // Kotlin coroutines & serialization
    implementation(libs.kotlinx.coroutines.core)
    implementation(libs.kotlinx.serialization.json)
}
