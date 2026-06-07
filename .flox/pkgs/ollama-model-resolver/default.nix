{
  lib,
  rustPlatform,
}:

rustPlatform.buildRustPackage rec {
  pname = "ollama-model-resolver";
  version = "0.1.0";

  src = ../../..;

  cargoLock.lockFile = "${src}/Cargo.lock";

  doCheck = true;

  meta = {
    description = "Resolve the best Ollama model variant for local hardware";
    license = lib.licenses.mit;
    mainProgram = "ollama-model-resolver";
    platforms = lib.platforms.linux;
  };
}
