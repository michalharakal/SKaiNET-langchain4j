plugins {
    alias(libs.plugins.kotlin.jvm)
}

kotlin {
    jvmToolchain(21)
}

dependencies {
    implementation(project(":langchain4j-skainet"))
    implementation(libs.langchain4j)
    implementation(libs.logback.classic)
}

val jvmArgs = listOf(
    "--add-modules=jdk.incubator.vector",
    "--enable-native-access=ALL-UNNAMED"
)

tasks.register<JavaExec>("runChat") {
    group = "examples"
    description = "Run the basic chat model example"
    classpath = sourceSets["main"].runtimeClasspath
    mainClass.set("sk.ainet.langchain4j.examples.SkainetChatModelExamplesKt")
    jvmArgs(jvmArgs)
}

tasks.register<JavaExec>("runStream") {
    group = "examples"
    description = "Run the streaming chat model example"
    classpath = sourceSets["main"].runtimeClasspath
    mainClass.set("sk.ainet.langchain4j.examples.SkainetStreamingChatModelExamplesKt")
    jvmArgs(jvmArgs)
}

tasks.register<JavaExec>("runFunctions") {
    group = "examples"
    description = "Run the function calling example"
    classpath = sourceSets["main"].runtimeClasspath
    mainClass.set("sk.ainet.langchain4j.examples.SkainetFunctionCallingExamplesKt")
    jvmArgs(jvmArgs)
}

tasks.register<JavaExec>("runRag") {
    group = "examples"
    description = "Run the RAG embedding example"
    classpath = sourceSets["main"].runtimeClasspath
    mainClass.set("sk.ainet.langchain4j.examples.SkainetBasicRagEmbedExamplesKt")
    jvmArgs(jvmArgs)
}
