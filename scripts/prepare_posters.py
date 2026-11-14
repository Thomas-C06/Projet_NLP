"""Cache public film posters locally for the cinema interface."""

import asyncio
from html.parser import HTMLParser
import json

import httpx

from backend.config import ROOT


class PosterParser(HTMLParser):
    def __init__(self):
        super().__init__()
        self.images = []

    def handle_starttag(self, tag, attrs):
        attributes = dict(attrs)
        if tag == "meta" and attributes.get("property") == "og:image":
            url = attributes.get("content", "")
            if url.startswith(("https://media.themoviedb.org/", "https://image.tmdb.org/")):
                self.images.append(url)


async def main():
    films = json.loads((ROOT / "data/movies.json").read_text(encoding="utf-8"))
    output = ROOT / "frontend/public/posters"
    output.mkdir(parents=True, exist_ok=True)
    manifest_path = output / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.exists() else {}
    semaphore = asyncio.Semaphore(2)
    failures = []
    async with httpx.AsyncClient(follow_redirects=True, timeout=20) as client:
        async def download(film):
            movie_id = str(film["tmdb_id"])
            if movie_id in manifest and (output / f"{movie_id}.jpg").exists():
                return
            async with semaphore:
                await asyncio.sleep(0.2)
                try:
                    response = await client.get(film["source"])
                    response.raise_for_status()
                    parser = PosterParser()
                    parser.feed(response.text)
                    if not parser.images:
                        return
                    image = await client.get(parser.images[0])
                    image.raise_for_status()
                    if not image.headers.get("content-type", "").startswith("image/"):
                        return
                    (output / f"{movie_id}.jpg").write_bytes(image.content)
                    manifest[movie_id] = {"title": film["title"], "source": film["source"],
                                          "image_source": parser.images[0]}
                    if movie_id == "152601" and len(parser.images) > 1:
                        backdrop = await client.get(parser.images[1])
                        if backdrop.status_code == 200:
                            (output / "hero-backdrop.jpg").write_bytes(backdrop.content)
                    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
                except httpx.HTTPError as error:
                    failures.append({"id": movie_id, "error": type(error).__name__})
        await asyncio.gather(*(download(film) for film in films))
    print(f"Local posters ready: {len(manifest)}/{len(films)}")
    if failures:
        print("Unavailable requests:", failures[:5])


if __name__ == "__main__":
    asyncio.run(main())
