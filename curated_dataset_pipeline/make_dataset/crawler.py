import argparse
import json
import subprocess
from typing import List, NamedTuple

import requests
from tqdm import tqdm
import dotenv
import os
dotenv.load_dotenv(override=True) 
HEADERS = {
    "Authorization": f"{os.environ.get("GITHUB_API_KEY")}",
    "Accept": "application/vnd.github.v3+json",
}


class RepoMetadata(NamedTuple):
    pass


class Crawler:
    def __init__(self, num_pages: int):
        self.num_pages = num_pages

    def search_repo(self) -> List[RepoMetadata]:
        """Search top 1000 java repositories

        Returns:
            List[RepoMetadata]: Github repositories metadata
        """
        all_elements: List[RepoMetadata] = []
        for page in tqdm(
            range(self.num_pages), desc="Crawling pages"
        ):
            url = f"https://api.github.com/search/repositories?q=language:java+pushed:>2023-10-01&sort=star&order=desc&per_page=100&page={page}"
            response = requests.get(url, headers=HEADERS)
            if response.status_code != 200:
                print(response.status_code)
                print(response.text)
                break
            cur_page_elements = response.json()
            all_elements += cur_page_elements["items"]
            page += 1
        return all_elements

    def store_repo_metadata(
        self, repo_metadata: List[RepoMetadata], storage_url: str
    ):
        """Store repo metadata into a file

        Args:
            repo_metadata (List[RepoMetadata]): Github repositories metadata
            storage_url (str): File url to store
        """
        with open(storage_url, "w") as f:
            json.dump(repo_metadata, f)

    def load_repo_metadata(self, storage_url: str) -> List[RepoMetadata]:
        """Load repo metadata from a file

        Args:
            storage_url (str): File url to load

        Returns:
            List[RepoMetadata]: Github repositories metadata
        """
        with open(storage_url, "r") as f:
            repo_metadata = json.loads(f.read())
        return repo_metadata

    def get_repo_html_url(self, repo_metadata: List[RepoMetadata]) -> List[str]:
        """Get repo html url from repo metadata

        Args:
            repo_metadata (List[RepoMetadata]): Github repositories metadata

        Returns:
            List[str]: Github repositories html url
        """
        repo_urls: List[str] = [repo["html_url"] for repo in repo_metadata]
        return repo_urls

    def clone_repo(self, repo_urls: str, repo_storage_url: str):
        """Clone repo

        Args:
            repo_urls (List[str]): Github repositories html url
            repo_storage_url (str): File url to store
        """
        for repo_url in tqdm(repo_urls, desc="Cloning repositories"):
            owner, repo = repo_url.split("/")[-2:]
            cmd = f"""
            if [ ! -d "{repo_storage_url}/{owner}_{repo}" ]
            then
                mkdir "{repo_storage_url}/{owner}_{repo}"
                cd "{repo_storage_url}/{owner}_{repo}"
                git clone --depth 1 https://github.com/{owner}/{repo}.git
            fi
            """
            subprocess.run(cmd, shell=True)


def main(args):
    # Crawl repo metadata and store into a file
    crawler = Crawler(num_pages=args.num_pages)
    repo_metadata = crawler.search_repo()
    crawler.store_repo_metadata(repo_metadata, args.repos_info)
    repo_urls = crawler.get_repo_html_url(repo_metadata)
    
    # Clone repo
    crawler.clone_repo(repo_urls, args.dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-pages", dest="num_pages", type=int)
    parser.add_argument("--repo-info", dest="repos_info")
    parser.add_argument("--dir", dest="dir")
    args = parser.parse_args()
    main(args)
