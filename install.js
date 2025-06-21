module.exports = {
  run: [
    {
      method: "shell.run",
      params: {
        venv: "env",
        message: [
          "git config --global --add safe.directory '*'",
          "git clone git@github.com:Stability-AI/stable-audio-tools.git"
        ]
      }
    },
    {
      method: "shell.run",
      params: {
        path: "stable-audio-tools",
        venv: "../env",
        message: [
          "pip install .",
        ]
      }
    },
    {
      method: "shell.run",
      params: {
        venv: "env",
        message: [
          "pip uninstall -y torch torchaudio torchvision",
          "uv pip install -r requirements.txt",
          "uv pip install -U bitsandbytes"
        ]
      }
    },
    {
      method: "script.start",
      params: {
        uri: "torch.js",
        params: {
          venv: "env",
          // xformers: true   // uncomment this line if your project requires xformers
        }
      }
    },
    {
      method: "fs.link",
      params: {
        venv: "env"
      }
    }
  ]
}
