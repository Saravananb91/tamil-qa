"""
scraper/spider.py
-----------------
Scrapy spider to crawl Tamil news articles from Dinamalar.
Extracted articles are used to build the Tamil corpus for tokenizer training.

Usage
-----
    scrapy crawl tamil_news -o data/scraped_articles.json
"""

import scrapy
from configs.config import SCRAPER_START_URL


class TamilNewsSpider(scrapy.Spider):
    """Crawl Dinamalar and extract article titles and body text."""

    name = "tamil_news"
    custom_settings = {
        "ROBOTSTXT_OBEY":    True,
        "DOWNLOAD_DELAY":    1.5,           # polite crawl delay
        "CONCURRENT_REQUESTS": 4,
        "LOG_LEVEL":         "WARNING",
    }

    def start_requests(self):
        yield scrapy.Request(SCRAPER_START_URL, callback=self.parse)

    def parse(self, response):
        """Follow all article links found on the main page."""
        article_links = response.css("a.article-link::attr(href)").getall()
        for link in article_links:
            yield response.follow(link, callback=self.parse_article)

    def parse_article(self, response):
        """Extract title and full body text from an article page."""
        title   = response.css("h1.article-title::text").get(default="").strip()
        content = " ".join(response.css("div.article-content p::text").getall()).strip()

        if title and content:
            yield {
                "title":   title,
                "content": content,
            }
