// fm-generate.swift — REAL on-device text generation via Apple's FoundationModels
// (macOS 26+, Apple Intelligence enabled). This is Apple's SYSTEM language model —
// NOT a coreai-catalog .aimodel (those are vision models needing macOS 27's CoreAI
// framework). Text-only: FoundationModels on macOS 26 has no image input.
//
// Build:  swiftc -O -parse-as-library tools/fm-generate.swift -o tools/fm-generate
// Run:    tools/fm-generate --prompt "Describe the Apple Neural Engine in one sentence."
//
// Exit codes: 0 ok · 1 generation error · 3 model unavailable (stderr: FM_UNAVAILABLE:<reason>)
import Foundation
import FoundationModels

@main
struct FMGenerate {
    static func main() async {
        var prompt = ""
        let args = Array(CommandLine.arguments.dropFirst())
        var i = 0
        while i < args.count {
            if args[i] == "--prompt", i + 1 < args.count { prompt = args[i + 1]; i += 2; continue }
            i += 1
        }
        if prompt.isEmpty {
            FileHandle.standardError.write("usage: fm-generate --prompt <text>\n".data(using: .utf8)!)
            exit(2)
        }
        let model = SystemLanguageModel.default
        guard case .available = model.availability else {
            FileHandle.standardError.write("FM_UNAVAILABLE: \(model.availability)\n".data(using: .utf8)!)
            exit(3)
        }
        do {
            let session = LanguageModelSession()
            let reply = try await session.respond(to: prompt)
            print(reply.content)
        } catch {
            FileHandle.standardError.write("FM_ERROR: \(error)\n".data(using: .utf8)!)
            exit(1)
        }
    }
}
