import subprocess

def clone_repository(repo_url, local_dir):
    try:
        subprocess.run(['git', 'clone', repo_url, local_dir], check=True)
        print(f'Repository cloned successfully into {local_dir}')
    except subprocess.CalledProcessError as e:
        print(f'Error occurred: {str(e)}')

# Replace with your repository URL and local directory
clone_repository('https://github.com/pcm-dpc/COVID-19/tree/master/dati-regioni', 'Italy_data')
