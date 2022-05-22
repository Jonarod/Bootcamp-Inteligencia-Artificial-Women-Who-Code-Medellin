# Usando a Docker

```
docker pull tensorflow/serving
```

```
docker run -p 8501:8501 --name tfserving_classifier \
  --mount type=bind,source=/path/to/the/unzipped/model/tmp/,target=/models/folder_name \
  -e MODEL_NAME=folder_name -t tensorflow/serving
```

por ejemplo
```
docker run -p 8501:8501 --name tfserving_classifier \
--mount type=bind,source=/Users/tf-server/img_classifier/,target=/models/img_classifier \
-e MODEL_NAME=img_classifier -t tensorflow/serving
```