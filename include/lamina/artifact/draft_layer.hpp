// include/lamina/artifact/draft_layer.hpp - Qwen3.8-Flash-Next's draft layer as a file: the GGUF tools/mtp_rt.py
// writes and the engine reads (--drafter FILE), its name where lamina-server looks for it, and where it comes from.
#pragma once

namespace lamina {

inline constexpr const char* kDraftLayerArch = "qwen4exp-mtp";   ///< its general.architecture
inline constexpr unsigned kDraftLayerVersion = 1;                 ///< its lamina.mtp.version: the layout the engine reads
inline constexpr const char* kDraftLayerFile = "Lamina-Qwen3.8-Flash-Next-MTP.gguf";
inline constexpr const char* kDraftLayerUrl = "https://huggingface.co/eddoursul/Lamina-Qwen3.8-Flash-Next-MTP";
/// this platform's script that downloads it into a folder
#ifdef _WIN32
inline constexpr const char* kDraftLayerScript = "download-draft-layer.bat";
#else
inline constexpr const char* kDraftLayerScript = "download-draft-layer.sh";
#endif

}  // namespace lamina
