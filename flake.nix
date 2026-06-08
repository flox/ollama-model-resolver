{
  description = "Dev shell for ollama-model-resolver (cargo build/run/test)";

  inputs.nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";

  outputs = { self, nixpkgs }:
    let
      systems = [ "aarch64-darwin" "x86_64-darwin" "aarch64-linux" "x86_64-linux" ];
      forAllSystems = f: nixpkgs.lib.genAttrs systems (system: f nixpkgs.legacyPackages.${system});
    in
    {
      devShells = forAllSystems (pkgs: {
        default = pkgs.mkShell {
          # Rust toolchain + dev tooling. mkShell's default stdenv provides the
          # C compiler and (on macOS) the Apple SDK that cargo's linker needs.
          packages = [
            pkgs.cargo
            pkgs.rustc
            pkgs.rust-analyzer
            pkgs.clippy
            pkgs.rustfmt
          ]
          # The linker needs libiconv on Darwin (the resolver links `-liconv`).
          ++ pkgs.lib.optionals pkgs.stdenv.isDarwin [ pkgs.libiconv ];

          # So rust-analyzer can find the standard library source.
          RUST_SRC_PATH = "${pkgs.rustPlatform.rustLibSrc}";

          shellHook = ''
            echo "ollama-model-resolver dev shell — $(cargo --version)"
            echo "Run: cargo run -- info   (needs a local Ollama; see README)"
          '';
        };
      });
    };
}
