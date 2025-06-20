module.exports = {
  run: [{
    method: "shell.run",
    params: {
      message: "git pull"
    }
  }, {
    method: "shell.run",
    params: {
      path: "stable-audio-tools",
      message: "git pull"
    }
  }, {
    method: "fs.rm",
    params: {
      path: "env"
    }
  }, {
    method: "shell.run",
    params: {
      path: "stable-audio-tools",
      venv: "../env",
      message: [
        "pip install .",
      ]
    }
  }, {
    method: "shell.run",
    params: {
      venv: "env",
      message: [
        "pip uninstall -y torch torchaudio torchvision",
        "uv pip install -r requirements.txt",
      ]
    }
  }, {
    method: "script.start",
    params: {
      uri: "torch.js",
      params: {
        venv: "env",
        // xformers: true   // uncomment this line if your project requires xformers
      }
    }
  }, {
    method: "fs.link",
    params: {
      venv: "env"
    }
  }]
}
