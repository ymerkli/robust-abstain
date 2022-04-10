# Download links (links expire, may need to be replaced)
anno='https://s3-eu-west-1.amazonaws.com/mapillary-data-research/mtsd/fully_annotated/mtsd_v2_fully_annotated_annotation.zip?AWSAccessKeyId=AKIAR47SN3BMGLPVNX36&Expires=1616330012&Signature=WVhCiuM%2FsIcuXo%2FaGQdVZJQKLSQ%3D'
test='https://s3-eu-west-1.amazonaws.com/mapillary-data-research/mtsd/fully_annotated/mtsd_v2_fully_annotated_images.test.zip?AWSAccessKeyId=AKIAR47SN3BMGLPVNX36&Expires=1616330012&Signature=euevPObZOnBag%2FaMTUzFer0t%2BDM%3D'
train0='https://s4-eu-west-1.amazonaws.com/mapillary-data-research/mtsd/fully_annotated/mtsd_v2_fully_annotated_images.train.0.zip?AWSAccessKeyId=AKIAR47SN3BMGLPVNX36&Expires=1616330012&Signature=DPqioY0uT3tXaq4kKYd6j8ync5s%3D'
train1='https://s3-eu-west-1.amazonaws.com/mapillary-data-research/mtsd/fully_annotated/mtsd_v2_fully_annotated_images.train.1.zip?AWSAccessKeyId=AKIAR47SN3BMGLPVNX36&Expires=1616330012&Signature=w5hwFtOCr7QopYzEw7RSjWKM%2BE8%3D'
train2='https://s3-eu-west-1.amazonaws.com/mapillary-data-research/mtsd/fully_annotated/mtsd_v2_fully_annotated_images.train.2.zip?AWSAccessKeyId=AKIAR47SN3BMGLPVNX36&Expires=1616330012&Signature=0F4Z3lNQ3Hk9rpNj26tLn94yg10%3D'
val='https://s3-eu-west-1.amazonaws.com/mapillary-data-research/mtsd/fully_annotated/mtsd_v2_fully_annotated_images.val.zip?AWSAccessKeyId=AKIAR47SN3BMGLPVNX36&Expires=1616330012&Signature=UxonjEeDtX4PAB7eutgtFBT1RN0%3D'

# Download Mapillary Traffic Sign Dataset
wget -O mtsd_v2_fully_annotated_annotation.zip "$anno" -P ./
wget -O mtsd_v2_fully_annotated_images.test.zip "$test" -P ./
wget -O mtsd_v2_fully_annotated_images.train.0.zip "$train0" -P ./
wget -O mtsd_v2_fully_annotated_images.train.1.zip "$train1" -P ./
wget -O mtsd_v2_fully_annotated_images.train.2.zip "$train2" -P ./
wget -O mtsd_v2_fully_annotated_images.val.zip "$val" -P ./

# Unzip it all
unzip mtsd_v2_fully_annotated_annotation.zip -d ./
unzip mtsd_v2_fully_annotated_images.test.zip -d ./
unzip mtsd_v2_fully_annotated_images.train.0.zip -d ./
unzip mtsd_v2_fully_annotated_images.train.1.zip -d ./
unzip mtsd_v2_fully_annotated_images.train.2.zip -d ./
unzip mtsd_v2_fully_annotated_images.val.zip -d ./

# Move to folder
mv images/ mtsd/
mv mtsd_v2_fully_annotated/* mtsd/

# Create directories
mkdir mtsd/images/train
mkdir mtsd/images/test
mkdir mtsd/images/val

# Prepare data
python mtsd_preprocessing.py