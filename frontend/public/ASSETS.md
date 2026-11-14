# Asset provenance

`tmdb-logo.svg` is an unmodified approved TMDB logo downloaded from:

https://www.themoviedb.org/assets/v4/logos/v2/blue_short-8e7b30f73a4020692ccca9c88bafe5dcb6f8a62a4c6bc55cd9ba82bb2cd95f6c.svg

Attribution and permitted use: https://www.themoviedb.org/about/logos-attribution

The TMDB brand belongs to its owner. Its use here identifies the film data source and does not imply endorsement.

## Film posters

`posters/` contains unmodified poster images cached from public TMDB film pages for the local educational application. `posters/manifest.json` records each film page and exact image source. These images are used with TMDB attribution and remain subject to the rights of their respective owners. The local files let the interface work offline. Run `python -m scripts.prepare_posters` to restore missing assets.

## Typography

`fonts/Manrope.ttf` and its `OFL.txt` license come from the Google Fonts Manrope repository:

https://github.com/google/fonts/tree/main/ofl/manrope

The font is served locally under the SIL Open Font License included with the asset.
