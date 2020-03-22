import os
import random


base_dir = "/cryosat3/btozer/CREATE_ML_FEATURES/tsv_all"
regions = ['AGSO', 'JAMSTEC', 'NGA', 'NGDC', 'NOAA_geodas', 'SIO', 'US_multi']

# AGSO   => .tsv_ky 
# others => .tsv
for region in regions:
    dirname = os.path.join(base_dir, region)
    ext = ".tsv"
    if region == "AGSO":
        ext = ".tsv_ky"
    filenames = [filename for filename in os.listdir(dirname) if filename.endswith(ext)]
    random.shuffle(filenames)
    filenames = [os.path.join(dirname, filename) for filename in filenames]

    s0, s1 = int(len(filenames) * 0.15), int(len(filenames) * 0.30)
    tests, validates, trains = filenames[:s0], filenames[s0:s1], filenames[s1:]
    for name, dataset in [("test", tests), ("validate", validates), ("train", trains)]:
        with open("{}-{}.txt".format(region, name), "w") as f:
            f.write("\n".join(dataset))

