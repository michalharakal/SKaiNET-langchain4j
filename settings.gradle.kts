pluginManagement {
    repositories {
        google()
        mavenCentral()
        gradlePluginPortal()
    }
}

dependencyResolutionManagement {
    repositories {
        google()
        mavenCentral()
    }
}

rootProject.name = "langchain-skainet"

// Include SKaiNET as a composite build with dependency substitution
includeBuild("../langchain_jlama/SKaiNET") {
    dependencySubstitution {
        substitute(module("sk.ainet:skainet-kllama")).using(project(":skainet-apps:skainet-kllama"))
        substitute(module("sk.ainet:skainet-kllama-agent")).using(project(":skainet-apps:skainet-kllama-agent"))
        substitute(module("sk.ainet:skainet-bert")).using(project(":skainet-apps:skainet-bert"))
        substitute(module("sk.ainet:skainet-llm")).using(project(":skainet-apps:skainet-llm"))
        substitute(module("sk.ainet:skainet-backend-cpu")).using(project(":skainet-backends:skainet-backend-cpu"))
        substitute(module("sk.ainet:skainet-lang-core")).using(project(":skainet-lang:skainet-lang-core"))
        substitute(module("sk.ainet:skainet-lang-ksp-annotations")).using(project(":skainet-lang:skainet-lang-ksp-annotations"))
        substitute(module("sk.ainet:skainet-compile-core")).using(project(":skainet-compile:skainet-compile-core"))
        substitute(module("sk.ainet:skainet-io-core")).using(project(":skainet-io:skainet-io-core"))
        substitute(module("sk.ainet:skainet-io-gguf")).using(project(":skainet-io:skainet-io-gguf"))
        substitute(module("sk.ainet:skainet-io-safetensors")).using(project(":skainet-io:skainet-io-safetensors"))
    }
}

include("langchain4j-skainet")
include("langchain4j-skainet-examples")
