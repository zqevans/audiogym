def get_custom_metadata(info, audio):

    relpath = info["relpath"]

    print(relpath)
    # Use relative path as the prompt
    return {"prompt": relpath}