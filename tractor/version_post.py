if True:
    version = git_version
    if version.startswith('main-gpu-'):
        version = '0.' + version.replace('main-gpu-', '')
    words = version.split('-')
    if len(words) == 3:
        version = words[0] + '.dev' + words[1]
    if version.startswith('dr'):
        version = version[2:]
