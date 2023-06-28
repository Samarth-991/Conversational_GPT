# Using the API


### 1. Installing the packages

Create an environment with the yml file:
```
conda env create --name environment_name -f environment.yml
```

OR

Updating an existing environment with the yml file:
```
conda env update --name environment_name --file environment.yml
```


### 2. Starting the API server

Navigate to the directory containing api.py file. Here, `src/conversation_bot/`

Start the server using:
```
uvicorn api:api
```

**Note**:- You can add --reload to watch for changes

---
---


## API Reference

#### 1. Health Check of API

```http
  GET /
```


#### 2. Convert using whisper model

```http
  GET /whisper
```

Body: 

| Field     | Type     | Description                |
| :-------- | :------- | :------------------------- |
| `urls`    | `List[string]` | **Required**. List of urls |


### 3. Convert using hugging face model

```http
  GET /huggingface
```

Body:
| Parameter | Type     | Description                       |
| :-------- | :------- | :-------------------------------- |
| `urls`    | `List[string]` | **Required**. List of urls |


