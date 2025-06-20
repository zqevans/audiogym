module.exports = {
  run: [{
    method: "fs.rm",
    params: {
      path: "stable-audio-tools"
    }
  }, {
    method: "fs.rm",
    params: {
      path: "env"
    }
  }]
}
