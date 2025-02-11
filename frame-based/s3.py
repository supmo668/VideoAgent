import boto3

class S3Module:
    def __init__(self, aws_access_key_id, aws_secret_access_key, region_name='us-east-1'):
        """
        Initialize the S3 client.

        :param aws_access_key_id: AWS Access Key ID
        :param aws_secret_access_key: AWS Secret Access Key
        :param region_name: AWS region (default: us-east-1)
        """
        self.s3_client = boto3.client(
            's3',
            aws_access_key_id=aws_access_key_id,
            aws_secret_access_key=aws_secret_access_key,
            region_name=region_name
        )

    def upload_file(self, file_path, bucket_name, object_name, acl=None):
        """
        Upload a file to an S3 bucket.

        :param file_path: Path to the local file to upload
        :param bucket_name: Name of the S3 bucket
        :param object_name: Name of the object in S3 (can include folder path)
        :param acl: Optional ACL (e.g., 'public-read')
        :return: True if upload succeeded, False otherwise
        """
        try:
            extra_args = {'ACL': acl} if acl else {}
            self.s3_client.upload_file(file_path, bucket_name, object_name, ExtraArgs=extra_args)
            print(f"File uploaded successfully to s3://{bucket_name}/{object_name}")
            return True
        except Exception as e:
            print(f"Error uploading file: {e}")
            return False

    def download_file(self, bucket_name, object_name, download_path):
        """
        Download a file from an S3 bucket.

        :param bucket_name: Name of the S3 bucket
        :param object_name: Name of the object in S3 (can include folder path)
        :param download_path: Local path to save the downloaded file
        :return: True if download succeeded, False otherwise
        """
        try:
            self.s3_client.download_file(bucket_name, object_name, download_path)
            print(f"File downloaded successfully from s3://{bucket_name}/{object_name} to {download_path}")
            return True
        except Exception as e:
            print(f"Error downloading file: {e}")
            return False