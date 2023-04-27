if True:
    words = git_version.split('-')
    if len(words) == 3:
        version = words[0] + '.dev' + words[1]
    else:
        version = git_version
    if version.startswith('dr'):
        version = version[2:]
