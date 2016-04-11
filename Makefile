
all: tracking_pb2.py

tracking_pb2.py:
	protoc -I=. --python_out=. tracking.proto

