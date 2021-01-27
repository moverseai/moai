def load_rgbd_skip(filename: str, folder: str):
    f = open(filename)
    f_lines = f.readlines()
    skip = 0
    for line in f_lines:
        if (folder in line):
            skip = int(line.split('\t')[1])
            break
    return skip