{
  description = "LLM study lab";

  inputs.nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";

  outputs = { nixpkgs, ... }:
    let
      pkgs = nixpkgs.legacyPackages.x86_64-linux;
    in {
      devShells.x86_64-linux.default = pkgs.mkShell {
        packages = [ pkgs.python313 pkgs.uv ];
        UV_PYTHON_DOWNLOADS = "never";
        UV_PYTHON = "${pkgs.python313}/bin/python3";
        JAX_PLATFORMS = "cuda";
        XLA_PYTHON_CLIENT_PREALLOCATE = "false";
        LD_LIBRARY_PATH = "${pkgs.lib.makeLibraryPath [
          pkgs.stdenv.cc.cc.lib
          pkgs.zlib
        ]}:/run/opengl-driver/lib";
      };
    };
}
