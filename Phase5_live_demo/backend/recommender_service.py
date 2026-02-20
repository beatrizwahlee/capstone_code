"""
recommender_service.py — ML wrapper with automatic mock fallback.

Real mode: loads BaselineRecommender + DiversityReranker + EchoChamberAnalyzer.
Mock mode: serves ~144 hard-coded articles across 12 MIND categories.
"""

from __future__ import annotations

import math
import random
import sys
import logging
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Mock article corpus (12 categories × 12 articles = 144 articles)
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Category aliases — maps quiz topic IDs to real MIND dataset categories.
# The MIND dataset does not have standalone 'science', 'technology', or
# 'politics' categories; those topics are covered under 'news'.
# ---------------------------------------------------------------------------

_CATEGORY_ALIASES: dict[str, list[str]] = {
    "science":    ["news"],
    "technology": ["news"],
    "politics":   ["news"],
}

MOCK_ARTICLES: list[dict] = [
    # ── SPORTS ──────────────────────────────────────────────────────────────
    {"news_id": "N001", "title": "Premier League: Manchester City's Haaland Scores Hat-Trick in Thrilling Victory", "category": "sports", "subcategory": "soccer", "abstract": "Manchester City secured a dominant 4-1 win over Arsenal on Saturday as Erling Haaland netted his tenth hat-trick of the season. The Norwegian striker is on pace to shatter all-time Premier League scoring records. Manager Pep Guardiola called it 'a perfect performance from the whole squad.'", "score": 0.95},
    {"news_id": "N002", "title": "Super Bowl Preview: Chiefs vs Eagles Clash in Epic Championship Showdown", "category": "sports", "subcategory": "nfl", "abstract": "Sunday's Super Bowl pits Patrick Mahomes and the Kansas City Chiefs against Jalen Hurts and the Philadelphia Eagles in what analysts are calling the most evenly matched championship in recent memory. Both teams enter the game on six-game winning streaks.", "score": 0.93},
    {"news_id": "N003", "title": "LeBron James Passes Record as Lakers Dominate West Coast Road Trip", "category": "sports", "subcategory": "nba", "abstract": "LeBron James surpassed Kareem Abdul-Jabbar's all-time scoring record last night, finishing with 38 points in a 112-98 Los Angeles Lakers victory. The achievement sparked a standing ovation from the opposing team's home crowd.", "score": 0.91},
    {"news_id": "N004", "title": "Djokovic Claims Record 25th Grand Slam Title at Australian Open", "category": "sports", "subcategory": "tennis", "abstract": "Novak Djokovic extended his unparalleled Grand Slam record with a clinical three-set victory at Melbourne Park. The Serbian legend showed no signs of slowing down despite being the oldest finalist in tournament history.", "score": 0.89},
    {"news_id": "N005", "title": "Yankees Sign Star Pitcher to Record $400 Million Contract Extension", "category": "sports", "subcategory": "baseball", "abstract": "The New York Yankees have locked up their ace pitcher for another decade in the largest contract ever given to a starting pitcher in MLB history. The deal includes a full no-trade clause and performance bonuses.", "score": 0.87},
    {"news_id": "N006", "title": "Tiger Woods Makes Emotional Return at Augusta National Masters Tournament", "category": "sports", "subcategory": "golf", "abstract": "Tiger Woods sent Augusta National into rapturous applause as he made the cut at the Masters for the first time since his career-threatening car accident. Woods shot a 68 in the second round to sit five shots off the lead.", "score": 0.85},
    {"news_id": "N007", "title": "Maple Leafs End 57-Year Cup Drought in Dramatic Game 7 Victory", "category": "sports", "subcategory": "hockey", "abstract": "The Toronto Maple Leafs finally ended their Stanley Cup championship drought with an overtime win in Game 7 against the Colorado Avalanche. The city erupted in celebration as Auston Matthews scored the series-winning goal.", "score": 0.83},
    {"news_id": "N008", "title": "USA Tops Medal Table as Paris Olympics Sets New Viewership Records", "category": "sports", "subcategory": "olympics", "abstract": "The United States led the final Olympics medal tally with 126 total medals, while the Paris Games broke streaming viewership records with 3.2 billion viewers globally. Several world records were shattered across athletics and swimming events.", "score": 0.81},
    {"news_id": "N009", "title": "Champions League: Real Madrid Edges Past Bayern Munich in Thriller Semi-Final", "category": "sports", "subcategory": "soccer", "abstract": "Real Madrid booked their place in yet another Champions League final with a nervy 3-2 aggregate victory over Bayern Munich. Vinicius Junior scored the decisive away goal in injury time to send the Bernabeu into hysteria.", "score": 0.79},
    {"news_id": "N010", "title": "Patrick Mahomes Sets Single Season Passing Yards Record in Week 14", "category": "sports", "subcategory": "nfl", "abstract": "Kansas City Chiefs quarterback Patrick Mahomes broke the single-season passing yards record with six weeks still remaining in the regular season. His 5,477 yards surpasses the previous mark set by Peyton Manning in 2013.", "score": 0.77},
    {"news_id": "N011", "title": "Golden State Warriors Unveil Superstar Lineup for Championship Push", "category": "sports", "subcategory": "nba", "abstract": "The Golden State Warriors have assembled what analysts are calling their most talented roster in franchise history after landing two All-Stars in the trade deadline. Head coach Steve Kerr says the team is hungry for another ring.", "score": 0.75},
    {"news_id": "N012", "title": "Serena Williams Foundation Announces $50M Tennis Scholarship Program", "category": "sports", "subcategory": "tennis", "abstract": "Serena Williams has pledged $50 million to fund tennis scholarships for underprivileged youth across 40 countries. The program aims to identify and develop 10,000 young talents over the next decade.", "score": 0.73},

    # ── HEALTH ──────────────────────────────────────────────────────────────
    {"news_id": "N013", "title": "FDA Approves Breakthrough Alzheimer's Drug That Slows Disease Progression", "category": "health", "subcategory": "medicine", "abstract": "The FDA has granted full approval to a new drug that clinical trials showed can slow Alzheimer's disease progression by 35% in early-stage patients. The treatment works by clearing amyloid plaques from the brain and is expected to reach pharmacies within six months.", "score": 0.94},
    {"news_id": "N014", "title": "New Study: 15 Minutes of Daily Exercise Significantly Extends Lifespan", "category": "health", "subcategory": "fitness", "abstract": "A landmark 20-year study tracking 100,000 participants found that just 15 minutes of moderate exercise per day reduces all-cause mortality by 22%. Researchers say even low-intensity walking counts, making the findings accessible to people of all fitness levels.", "score": 0.92},
    {"news_id": "N015", "title": "Mediterranean Diet Ranked World's Best for Seventh Consecutive Year", "category": "health", "subcategory": "diet", "abstract": "U.S. News & World Report has once again crowned the Mediterranean diet the world's best overall diet. Nutritionists credit its emphasis on olive oil, fish, whole grains and vegetables for reducing heart disease risk and promoting longevity.", "score": 0.90},
    {"news_id": "N016", "title": "Mental Health Crisis Among Teens: New Digital Therapy Shows Promise", "category": "health", "subcategory": "mentalhealth", "abstract": "A randomized controlled trial found that a new AI-powered therapy app reduced depression symptoms in teenagers by 40% over eight weeks. Mental health experts cautiously welcomed the findings while emphasizing it should complement, not replace, professional care.", "score": 0.88},
    {"news_id": "N017", "title": "Universal Flu Vaccine Shows 95% Effectiveness in Phase 3 Trials", "category": "health", "subcategory": "vaccines", "abstract": "Scientists have announced that a new universal influenza vaccine provided protection against all known flu strains in a 50,000-person clinical trial. If approved, the vaccine could eliminate the need for annual flu shots and prevent millions of hospitalizations.", "score": 0.86},
    {"news_id": "N018", "title": "Scientists Discover Protein That Could Reverse Cellular Aging", "category": "health", "subcategory": "research", "abstract": "Researchers at Stanford University have identified a protein that reverses key markers of cellular aging in mice, extending their healthy lifespan by 30%. Human trials are expected to begin within two years, raising cautious optimism about longevity treatments.", "score": 0.84},
    {"news_id": "N019", "title": "Top Trainers Share Their Favorite HIIT Workouts for Maximum Results", "category": "health", "subcategory": "fitness", "abstract": "Elite personal trainers and sports scientists reveal the most effective high-intensity interval training protocols for fat loss and cardiovascular fitness. Their consensus: four 20-minute sessions per week outperform longer, steady-state cardio.", "score": 0.82},
    {"news_id": "N020", "title": "Coffee Drinkers Live Longer According to Massive 20-Year Health Study", "category": "health", "subcategory": "diet", "abstract": "A sweeping new study of 500,000 people across 10 countries found that drinking three to five cups of coffee per day is associated with a 12% lower risk of all-cause mortality. Researchers point to coffee's high antioxidant content as a key factor.", "score": 0.80},
    {"news_id": "N021", "title": "New Cancer Detection Blood Test Catches Disease Five Years Earlier", "category": "health", "subcategory": "medicine", "abstract": "A revolutionary multi-cancer early detection blood test has demonstrated 94% accuracy in identifying cancer up to five years before symptoms appear. The test, which screens for 50 cancer types from a single blood draw, could transform oncology screening.", "score": 0.78},
    {"news_id": "N022", "title": "Meditation vs. Therapy: What the Latest Research Actually Shows", "category": "health", "subcategory": "mentalhealth", "abstract": "A comprehensive meta-analysis of 200 studies compares the effectiveness of mindfulness meditation with cognitive behavioral therapy for anxiety and depression. The findings suggest a combination approach is more effective than either treatment alone.", "score": 0.76},
    {"news_id": "N023", "title": "CRISPR Gene Therapy Successfully Treats Inherited Blindness in Trial", "category": "health", "subcategory": "research", "abstract": "In a landmark gene therapy trial, CRISPR editing restored functional vision in 14 out of 16 patients with a previously incurable inherited form of blindness. The treatment delivered a corrective gene directly into photoreceptor cells in the retina.", "score": 0.74},
    {"news_id": "N024", "title": "Remote Patient Monitoring Reduces Hospital Readmissions by 40%", "category": "health", "subcategory": "medicine", "abstract": "Hospitals deploying continuous remote monitoring wearables for post-discharge heart failure patients saw a 40% reduction in 30-day readmissions. The devices track heart rate, blood pressure and fluid retention, alerting clinicians to warning signs early.", "score": 0.72},

    # ── TECHNOLOGY ──────────────────────────────────────────────────────────
    {"news_id": "N025", "title": "OpenAI Releases Next-Generation Model With Human-Level Reasoning", "category": "technology", "subcategory": "ai", "abstract": "OpenAI's latest language model achieves human-level performance on a suite of professional reasoning benchmarks including the bar exam, medical licensing test, and mathematical olympiad. The model introduces a novel chain-of-thought architecture.", "score": 0.96},
    {"news_id": "N026", "title": "Major Data Breach Exposes 500 Million Users' Personal Information", "category": "technology", "subcategory": "cybersecurity", "abstract": "A sophisticated cyberattack on a major cloud services provider has compromised the personal data of 500 million users, including names, emails and encrypted passwords. Security researchers traced the breach to a zero-day vulnerability in the authentication system.", "score": 0.94},
    {"news_id": "N027", "title": "Apple Unveils Revolutionary AR Glasses Designed to Replace Smartphones", "category": "technology", "subcategory": "gadgets", "abstract": "Apple has announced its most ambitious product in a decade: a pair of augmented reality glasses that can display information, take calls, and run apps without a phone. The device, priced at $2,499, ships to developers in the spring.", "score": 0.92},
    {"news_id": "N028", "title": "Microsoft Integrates AI Copilot Into Every Windows Application", "category": "technology", "subcategory": "software", "abstract": "Microsoft's sweeping Windows update embeds an AI assistant directly into every first-party application, enabling users to automate tasks, generate content and analyze data with plain-language commands. Early reviews praise the productivity gains.", "score": 0.90},
    {"news_id": "N029", "title": "SpaceX Starlink Now Provides Internet Connectivity to 100 Countries", "category": "technology", "subcategory": "internet", "abstract": "SpaceX's Starlink satellite internet service has reached the milestone of 100 countries served, with over 5 million active subscribers worldwide. The company reports average download speeds of 150 Mbps even in remote and rural areas.", "score": 0.88},
    {"news_id": "N030", "title": "Google DeepMind Achieves New Milestone in Protein Structure Prediction", "category": "technology", "subcategory": "ai", "abstract": "DeepMind's AlphaFold 3 has predicted the 3D structures of virtually all known proteins with atomic accuracy, a breakthrough that scientists say will accelerate drug discovery by a decade. The complete database has been made freely available to researchers.", "score": 0.86},
    {"news_id": "N031", "title": "New Ransomware Group Targets Critical Infrastructure in 30 Countries", "category": "technology", "subcategory": "cybersecurity", "abstract": "Cybersecurity agencies in the US, UK and EU have issued a joint warning about a new ransomware collective that has attacked hospitals, power grids and water treatment facilities across 30 nations. Experts advise organizations to patch known vulnerabilities immediately.", "score": 0.84},
    {"news_id": "N032", "title": "Samsung Galaxy Ultra Sets New Benchmark for Smartphone Cameras", "category": "technology", "subcategory": "gadgets", "abstract": "DxOMark has awarded the new Samsung Galaxy Ultra the highest camera score in smartphone history, praising its 200-megapixel main sensor, 10x optical zoom, and AI-powered computational photography that rivals professional DSLR cameras.", "score": 0.82},
    {"news_id": "N033", "title": "Open Source AI Model Outperforms Commercial Alternatives in Benchmarks", "category": "technology", "subcategory": "software", "abstract": "A new open-source language model released by a coalition of academic researchers outscores GPT-4 on 12 of 15 standard benchmarks while running on consumer hardware. The model's open weights have been downloaded 2 million times in its first week.", "score": 0.80},
    {"news_id": "N034", "title": "Meta Announces Next-Generation Social VR Platform for 2025", "category": "technology", "subcategory": "internet", "abstract": "Meta has unveiled its most immersive virtual reality social platform yet, featuring photorealistic avatars, haptic feedback gloves and spatial audio that its developers say creates true presence. The platform will support up to 1,000 simultaneous users in a single space.", "score": 0.78},
    {"news_id": "N035", "title": "Self-Driving Cars Ready for Full Commercial Deployment, Tesla Claims", "category": "technology", "subcategory": "ai", "abstract": "Tesla CEO Elon Musk announced that the company's Full Self-Driving software has achieved regulatory approval for Level 4 autonomous operation in 12 US states. The rollout begins in Q2 with a fleet of robotaxis operating in major metropolitan areas.", "score": 0.76},
    {"news_id": "N036", "title": "Robot Chef Can Prepare 500 Different Recipes With Consistent Perfection", "category": "technology", "subcategory": "gadgets", "abstract": "A kitchen robotics startup has launched a countertop robot chef that can autonomously prepare 500 different recipes with restaurant-quality results. The device uses computer vision and precise motor control to chop, stir and plate meals in 20 minutes.", "score": 0.74},

    # ── POLITICS ────────────────────────────────────────────────────────────
    {"news_id": "N037", "title": "Historic Voter Turnout in Midterms Reshapes Congressional Landscape", "category": "politics", "subcategory": "elections", "abstract": "Record-breaking voter participation in the midterm elections has produced a dramatically altered Congress, with independent candidates winning 15 seats and shifting the balance of power. Political analysts say the results signal a historic rejection of extreme partisanship.", "score": 0.91},
    {"news_id": "N038", "title": "Senate Passes Landmark Climate Bill After Months of Bipartisan Talks", "category": "politics", "subcategory": "policy", "abstract": "The US Senate passed a sweeping climate bill 64-36 with unusual bipartisan support, allocating $1.2 trillion for clean energy transition, grid modernization and domestic manufacturing. The legislation is considered the most significant climate action in US history.", "score": 0.89},
    {"news_id": "N039", "title": "NATO Summit Agrees on Largest Defense Spending Increase in History", "category": "politics", "subcategory": "international", "abstract": "NATO member states at the Brussels summit unanimously agreed to raise defense spending targets to 3% of GDP, the alliance's largest coordinated military investment since the Cold War. The decision was driven by an evolving security landscape in Eastern Europe.", "score": 0.87},
    {"news_id": "N040", "title": "House Passes Comprehensive Immigration Reform Bill 240-190", "category": "politics", "subcategory": "congress", "abstract": "A bipartisan coalition in the House of Representatives passed the most comprehensive immigration reform package in 35 years, creating a pathway to citizenship for 11 million undocumented residents while strengthening border security measures.", "score": 0.85},
    {"news_id": "N041", "title": "President Signs Executive Order on AI Safety and National Security", "category": "politics", "subcategory": "policy", "abstract": "The President signed a sweeping executive order establishing mandatory safety evaluations for large AI models before commercial deployment, along with new rules restricting the export of AI chips to adversarial nations. Tech industry reaction has been mixed.", "score": 0.83},
    {"news_id": "N042", "title": "Poll Shows Independent Voters Shifting Away From Both Major Parties", "category": "politics", "subcategory": "elections", "abstract": "A new Gallup survey finds that 47% of Americans now identify as independent, an all-time high, with both parties seeing declining loyalty among voters under 40. Analysts warn the trend could upend traditional election modeling.", "score": 0.81},
    {"news_id": "N043", "title": "Supreme Court Rules in Landmark Digital Privacy Case", "category": "politics", "subcategory": "policy", "abstract": "In a 6-3 decision, the Supreme Court ruled that the government requires a warrant to access real-time location data from cell phones, dramatically expanding Fourth Amendment protections into the digital era. Legal experts are calling it a generational ruling.", "score": 0.79},
    {"news_id": "N044", "title": "G7 Leaders Announce Coordinated Strategy to Combat Inflation", "category": "politics", "subcategory": "international", "abstract": "The G7 summit concluded with a joint communiqué outlining a synchronized fiscal and monetary policy response to persistent global inflation. Leaders agreed on coordinated interest rate communication and a strategic reserve release to stabilize energy markets.", "score": 0.77},
    {"news_id": "N045", "title": "Debt Ceiling Deal Reached After Weeks of Tense Negotiations", "category": "politics", "subcategory": "congress", "abstract": "Congressional leaders and the White House reached a last-minute agreement to raise the debt ceiling, averting a first-ever US default. The deal includes $500 billion in spending caps over two years and new work requirements for some benefit programs.", "score": 0.75},
    {"news_id": "N046", "title": "New Federal Standards Prioritize AI and Digital Literacy in Schools", "category": "politics", "subcategory": "policy", "abstract": "The Department of Education released new voluntary national standards making artificial intelligence literacy and computational thinking core curriculum components from kindergarten through 12th grade, responding to a bipartisan push to prepare students for an AI-driven economy.", "score": 0.73},
    {"news_id": "N047", "title": "UN Climate Summit Reaches Historic Agreement on Carbon Reduction", "category": "politics", "subcategory": "international", "abstract": "Nations at the COP31 climate summit have agreed to a binding framework requiring 50% reduction in carbon emissions by 2035, with a $500 billion fund to help developing nations transition to clean energy. The deal represents the most ambitious climate commitment ever made.", "score": 0.71},
    {"news_id": "N048", "title": "White House Unveils $2 Trillion Infrastructure Renewal Initiative", "category": "politics", "subcategory": "policy", "abstract": "The administration announced the largest domestic infrastructure investment in American history, targeting roads, bridges, high-speed rail, broadband internet and water systems. The plan would be funded through a combination of bonds and targeted corporate tax increases.", "score": 0.69},

    # ── FINANCE ─────────────────────────────────────────────────────────────
    {"news_id": "N049", "title": "S&P 500 Hits Record High as Tech Earnings Surpass All Expectations", "category": "finance", "subcategory": "stocks", "abstract": "The S&P 500 index reached a new all-time high after the big technology companies reported quarterly earnings that handily beat analyst estimates. Nvidia's results were particularly stunning, with revenue up 265% year-over-year driven by insatiable AI chip demand.", "score": 0.93},
    {"news_id": "N050", "title": "Federal Reserve Holds Rates Steady Amid Mixed Economic Signals", "category": "finance", "subcategory": "economy", "abstract": "The Federal Reserve's Open Market Committee voted unanimously to hold the federal funds rate at its current level, citing continued labor market strength but noting early signs of slowing consumer spending. Markets interpreted the decision as a pause before potential cuts.", "score": 0.91},
    {"news_id": "N051", "title": "Bitcoin Surges Past $100,000 as Institutional Investors Pile In", "category": "finance", "subcategory": "crypto", "abstract": "Bitcoin crossed the historic $100,000 milestone for the first time, driven by record inflows into spot Bitcoin ETFs and corporate treasury adoption. BlackRock's Bitcoin fund has now surpassed $50 billion in assets under management.", "score": 0.89},
    {"news_id": "N052", "title": "Housing Market Cools Sharply as Mortgage Rates Hit 20-Year High", "category": "finance", "subcategory": "realestate", "abstract": "The US housing market is experiencing its sharpest slowdown in decades as 30-year fixed mortgage rates reached 7.8%, the highest since 2001. Existing home sales fell to their lowest monthly level in 13 years as affordability reaches crisis levels for first-time buyers.", "score": 0.87},
    {"news_id": "N053", "title": "Top Advisors Share Secrets to Retiring Comfortably at Age 50", "category": "finance", "subcategory": "personalfinance", "abstract": "Certified financial planners reveal the specific savings rates, investment allocations and lifestyle strategies that their clients who successfully retired before 55 used. The common thread: aggressive saving in the 30s combined with low-cost index fund investing.", "score": 0.85},
    {"news_id": "N054", "title": "Berkshire Hathaway Reveals Surprising New Billion-Dollar Investments", "category": "finance", "subcategory": "stocks", "abstract": "Warren Buffett's Berkshire Hathaway disclosed major new positions in a Japanese trading conglomerate, a US homebuilder, and a satellite technology company. The reveals sent those stocks surging as investors rushed to follow the Oracle of Omaha's lead.", "score": 0.83},
    {"news_id": "N055", "title": "Unemployment Falls to Historic Low as Job Market Stays Resilient", "category": "finance", "subcategory": "economy", "abstract": "The US unemployment rate fell to 3.2%, the lowest in 54 years, as the economy added 280,000 jobs last month, dramatically exceeding economist forecasts. Wage growth continues to outpace inflation for the third consecutive quarter.", "score": 0.81},
    {"news_id": "N056", "title": "SEC Approves First Spot Bitcoin ETF in Historic Regulatory Decision", "category": "finance", "subcategory": "crypto", "abstract": "The Securities and Exchange Commission granted approval to 11 spot Bitcoin exchange-traded funds, opening the cryptocurrency to millions of retirement account holders for the first time. On its first day, the combined products attracted $4.6 billion in inflows.", "score": 0.79},
    {"news_id": "N057", "title": "Commercial Real Estate Crisis Deepens as Office Vacancies Soar", "category": "finance", "subcategory": "realestate", "abstract": "US office vacancy rates have hit a post-war record of 19.6% as remote work permanently reshapes demand for commercial space. Several regional banks with heavy commercial real estate exposure are under scrutiny from regulators concerned about loan book quality.", "score": 0.77},
    {"news_id": "N058", "title": "How to Build a Million-Dollar Retirement Fund on an Average Salary", "category": "finance", "subcategory": "personalfinance", "abstract": "Personal finance experts break down the exact steps that allow someone earning the median US salary to accumulate $1 million by age 65 through disciplined saving, 401(k) matching, and compound growth. The math shows it requires saving just 15% of income from age 25.", "score": 0.75},
    {"news_id": "N059", "title": "AI Stocks Outperform the Broader Market by 200% in Stunning Bull Run", "category": "finance", "subcategory": "stocks", "abstract": "A basket of pure-play AI infrastructure stocks has generated 200% returns over the past 18 months, dwarfing the S&P 500's 28% gain over the same period. Analysts debate whether the rally reflects fundamentals or a speculative bubble reminiscent of 1999.", "score": 0.73},
    {"news_id": "N060", "title": "US Trade Deficit Narrows Sharply as Exports Reach Record Levels", "category": "finance", "subcategory": "economy", "abstract": "The monthly US trade deficit fell to its lowest level in six years as exports of semiconductor chips, LNG and agricultural products hit records. Economists credit the weak dollar and strong global demand for American technology products.", "score": 0.71},

    # ── ENTERTAINMENT ────────────────────────────────────────────────────────
    {"news_id": "N061", "title": "Avengers: Secret Wars Breaks Opening Weekend Box Office Record", "category": "entertainment", "subcategory": "movies", "abstract": "Marvel's Avengers: Secret Wars shattered every box office record with a $650 million opening weekend globally, surpassing the previous record set by Avengers: Endgame. Critics are calling it the most ambitious superhero film ever made.", "score": 0.94},
    {"news_id": "N062", "title": "Taylor Swift Drops Surprise Album and Extends Eras Tour Into 2026", "category": "entertainment", "subcategory": "music", "abstract": "Taylor Swift surprised fans with a midnight album release of 'The Tortured Poets Department Vol. 2' featuring 18 new tracks. She simultaneously announced 80 additional Eras Tour dates extending the record-breaking concert series into 2026.", "score": 0.92},
    {"news_id": "N063", "title": "Beyoncé Wins Record 10th Grammy for Album of the Year", "category": "entertainment", "subcategory": "awards", "abstract": "Beyoncé made Grammy history by winning her tenth Album of the Year trophy for 'Cowboy Carter,' breaking the record she shares with Adele. The ceremony also saw record TV ratings and multiple historic wins for first-time nominees.", "score": 0.90},
    {"news_id": "N064", "title": "Oscars 2025: Full Winners List From a Historic Ceremony", "category": "entertainment", "subcategory": "awards", "abstract": "The 97th Academy Awards produced a historic night with the first AI-assisted film winning Best Picture and the oldest Best Actress winner in Oscar history. The ceremony attracted the largest US television audience in a decade.", "score": 0.88},
    {"news_id": "N065", "title": "Netflix Drama Series Becomes Platform's Most-Watched Show in History", "category": "entertainment", "subcategory": "tv", "abstract": "Netflix's new political thriller series amassed 200 million views in its first two weeks, overtaking Squid Game as the platform's most-watched series. The show sparked widespread cultural conversation and multiple government responses to its themes.", "score": 0.86},
    {"news_id": "N066", "title": "Christopher Nolan's New Film Receives Standing Ovation at Cannes", "category": "entertainment", "subcategory": "movies", "abstract": "Christopher Nolan's Interstellar follow-up received a 12-minute standing ovation at the Cannes Film Festival, with critics calling it his masterpiece. The film explores artificial consciousness through the lens of a philosophical thriller.", "score": 0.84},
    {"news_id": "N067", "title": "Celebrity Power Couple Announce Shocking Split After Seven Years", "category": "entertainment", "subcategory": "celebrities", "abstract": "Hollywood's most beloved couple announced their separation in a joint statement requesting privacy. The split has dominated social media for 48 hours, with their respective fan bases rallying in support. Both will continue co-parenting their two children.", "score": 0.82},
    {"news_id": "N068", "title": "The Weeknd Announces Farewell Tour After 15 Iconic Years in Music", "category": "entertainment", "subcategory": "music", "abstract": "In a heartfelt open letter, The Weeknd announced a worldwide farewell tour before an indefinite break from music. The 'After Hours til Dawn' final world tour will visit 60 cities across six continents and is expected to gross $1 billion.", "score": 0.80},
    {"news_id": "N069", "title": "Grammy Nominations: Surprising Snubs and Unexpected Breakout Inclusions", "category": "entertainment", "subcategory": "awards", "abstract": "The Recording Academy released this year's Grammy nominations, which have already generated significant controversy for omitting several critically acclaimed artists while including viral TikTok musicians in major categories for the first time.", "score": 0.78},
    {"news_id": "N070", "title": "Pixar's New Film Earns Perfect 100% on Rotten Tomatoes at Release", "category": "entertainment", "subcategory": "movies", "abstract": "Pixar's latest original film has become only the third in history to open with a perfect Rotten Tomatoes score. The animated feature, which tackles themes of grief and memory, has moved both children and adults to tears in early screenings.", "score": 0.76},
    {"news_id": "N071", "title": "Viral TikTok Trend Drives 500% Surge in Vintage Clothing Sales", "category": "entertainment", "subcategory": "celebrities", "abstract": "A TikTok trend where influencers styled 1990s thrift store finds caused a massive spike in vintage clothing demand, with resale platforms reporting a 500% week-over-week increase in sales of Y2K fashion items.", "score": 0.74},
    {"news_id": "N072", "title": "Live Music Revenue Surpasses Pre-Pandemic Levels for First Time", "category": "entertainment", "subcategory": "music", "abstract": "The global live music industry generated $32 billion in revenue last year, surpassing pre-pandemic highs for the first time. Concert ticket prices remain elevated but so does demand, with stadium tours selling out within minutes of going on sale.", "score": 0.72},

    # ── TRAVEL ──────────────────────────────────────────────────────────────
    {"news_id": "N073", "title": "10 Hidden European Villages That Rival the Most Famous Tourist Spots", "category": "travel", "subcategory": "destinations", "abstract": "Travel writers reveal the lesser-known European gems offering the same medieval charm, local cuisine and scenic beauty as popular destinations but without the crowds. From Slovenia's Škofja Loka to Portugal's Monsaraz, these villages reward the adventurous traveler.", "score": 0.88},
    {"news_id": "N074", "title": "How to Fly Business Class for the Price of Economy Using Miles", "category": "travel", "subcategory": "tips", "abstract": "Frequent flyer experts reveal the specific credit card sign-up bonuses, transfer partner sweet spots and off-peak booking windows that reliably secure business class seats on transcontinental flights for under $100 in fees plus miles.", "score": 0.86},
    {"news_id": "N075", "title": "New Non-Stop Route Cuts New York to Tokyo Flight Time to 11 Hours", "category": "travel", "subcategory": "airlines", "abstract": "Japan Airlines has launched a new polar routing from JFK to Narita that saves 90 minutes versus traditional routes, enabled by new generation long-range aircraft. The route has already sold out its first six months of premium cabin seats.", "score": 0.84},
    {"news_id": "N076", "title": "World's First Underwater Hotel Suite Now Accepting Reservations", "category": "travel", "subcategory": "hotels", "abstract": "A luxury resort in the Maldives has opened a fully underwater suite that offers 360-degree views of a coral reef from the bedroom, living room and private dining area. The $25,000-per-night suite requires guests to descend via a spiral staircase below the water line.", "score": 0.82},
    {"news_id": "N077", "title": "Climbing Everest in 2025: New Regulations Change Everything", "category": "travel", "subcategory": "adventure", "abstract": "Nepal's government has introduced sweeping new regulations for Everest expeditions including mandatory guide ratios, updated health requirements and fixed climbing windows to reduce overcrowding on the mountain's notoriously dangerous upper sections.", "score": 0.80},
    {"news_id": "N078", "title": "Japan's Cherry Blossom Season 2025: Best Spots and Exact Dates", "category": "travel", "subcategory": "destinations", "abstract": "Meteorologists and the Japan Meteorological Corporation have released their annual cherry blossom forecast, predicting an early and spectacular hanami season. Tokyo's Ueno Park is expected to peak between March 24-28, with Kyoto's Maruyama Park following April 1-5.", "score": 0.78},
    {"news_id": "N079", "title": "Carry-On Only Travel: The Complete Guide to Packing Smarter", "category": "travel", "subcategory": "tips", "abstract": "Professional travel hackers and digital nomads share their tested system for fitting two weeks of clothing, electronics and toiletries into a single underseat carry-on bag, saving significant time and money on checked baggage fees.", "score": 0.76},
    {"news_id": "N080", "title": "Budget Airlines Offer Transatlantic Flights From $99 Starting March", "category": "travel", "subcategory": "airlines", "abstract": "Several ultra-low-cost carriers have announced a price war on transatlantic routes, offering seats from $99 between select US cities and Europe. Travel experts warn the prices come with strict luggage allowances and minimal onboard amenities.", "score": 0.74},
    {"news_id": "N081", "title": "Best Beach Resorts in Southeast Asia Under $100 Per Night", "category": "travel", "subcategory": "hotels", "abstract": "A comprehensive guide to the finest beachfront resorts across Thailand, Bali and the Philippines that offer private beach access, pools and quality dining for under $100 per night when booked directly. The list prioritizes properties rated 4.5 stars or above.", "score": 0.72},
    {"news_id": "N082", "title": "Northern Lights Tourism Booms as Solar Activity Peaks", "category": "travel", "subcategory": "destinations", "abstract": "Aurora borealis tourism has exploded to record levels as the sun approaches its 11-year solar maximum, making northern lights sightings more frequent and spectacular than any time since 2003. Iceland, Norway and Finland have seen 300% increases in winter bookings.", "score": 0.70},
    {"news_id": "N083", "title": "Solo Female Travel Safety Guide: Essential Tips for Every Destination", "category": "travel", "subcategory": "adventure", "abstract": "Experienced solo female travelers and safety experts share the practical precautions, booking strategies and local knowledge that make solo travel both empowering and secure in destinations from Southeast Asia to South America.", "score": 0.68},
    {"news_id": "N084", "title": "Best Travel Rewards Cards That Earn Free Flights Fastest in 2025", "category": "travel", "subcategory": "tips", "abstract": "Credit card analysts have ranked the top travel rewards cards based on sign-up bonuses, earning rates and transfer partner value. The top three cards can generate enough points for a business class round trip to Europe within three months of card opening.", "score": 0.66},

    # ── SCIENCE ─────────────────────────────────────────────────────────────
    {"news_id": "N085", "title": "NASA Artemis Mission Successfully Lands Astronauts on the Moon", "category": "science", "subcategory": "space", "abstract": "For the first time since Apollo 17 in 1972, NASA astronauts have walked on the lunar surface as part of the Artemis program. The mission, which included the first woman and first person of color to land on the Moon, marks the beginning of permanent lunar exploration.", "score": 0.97},
    {"news_id": "N086", "title": "Arctic Ice Sheet Hits Lowest Recorded Level in February", "category": "science", "subcategory": "climate", "abstract": "Scientists report that Arctic sea ice extent has reached its lowest February measurement since satellite records began, 1.2 million square kilometers below the 1981-2010 average. Climate researchers say the accelerating loss is reshaping global weather patterns.", "score": 0.95},
    {"news_id": "N087", "title": "New Deep-Sea Species Discovered in Unprecedented Pacific Expedition", "category": "science", "subcategory": "biology", "abstract": "A scientific expedition to the Mariana Trench using next-generation submersibles has discovered 27 previously unknown species, including a bioluminescent shrimp, a transparent octopus and a bacteria that thrives at pressures 1,000 times greater than sea level.", "score": 0.93},
    {"news_id": "N088", "title": "Physicists Achieve Room-Temperature Superconductivity Breakthrough", "category": "science", "subcategory": "physics", "abstract": "Researchers at MIT have confirmed a material that conducts electricity with zero resistance at room temperature and normal atmospheric pressure. If the findings are replicated, the discovery would transform power transmission, computing and transportation.", "score": 0.91},
    {"news_id": "N089", "title": "Amazon Reforestation Project Plants Billionth Tree in Record Time", "category": "science", "subcategory": "environment", "abstract": "A coalition of NGOs, local communities and tech companies reached the milestone of planting one billion native trees across 2.4 million hectares of the Brazilian Amazon, three years ahead of schedule. Satellite data confirms a measurable increase in regional rainfall.", "score": 0.89},
    {"news_id": "N090", "title": "Webb Telescope Finds Possible Biosignatures on Habitable Exoplanet", "category": "science", "subcategory": "space", "abstract": "The James Webb Space Telescope has detected dimethyl sulfide in the atmosphere of K2-18b, a chemical that on Earth is only produced by living organisms. Scientists urge caution but say the finding is the most promising hint of extraterrestrial biology ever detected.", "score": 0.87},
    {"news_id": "N091", "title": "Record Heatwaves Conclusively Linked to Climate Change", "category": "science", "subcategory": "climate", "abstract": "An authoritative new attribution study finds that the devastating heatwaves of last summer were made at least 5 times more likely and 3°C hotter by human-caused climate change. The findings, from 27 research institutions, represent scientific consensus.", "score": 0.85},
    {"news_id": "N092", "title": "Gene Drive Eliminates Malaria-Carrying Mosquitoes in Field Trial", "category": "science", "subcategory": "biology", "abstract": "The first open-environment test of a gene drive successfully reduced populations of Anopheles malaria mosquitoes by 95% across a 100 square-kilometer area in Burkina Faso, with no detectable effect on other insect species in the ecosystem.", "score": 0.83},
    {"news_id": "N093", "title": "Quantum Internet Transmits Secure Data Over 1,000 Kilometers", "category": "science", "subcategory": "physics", "abstract": "Chinese scientists have demonstrated quantum-encrypted data transmission over 1,000 kilometers using a combination of ground-based fiber and a quantum satellite relay, a critical step toward a globally secure quantum internet.", "score": 0.81},
    {"news_id": "N094", "title": "SpaceX Mars Mission First Human Landing Planned for 2028", "category": "science", "subcategory": "space", "abstract": "SpaceX CEO Elon Musk presented updated plans for the first crewed Mars landing mission, targeting a launch in late 2027 for a 2028 surface landing. The mission will use a fully reusable Starship configuration carrying six astronauts and 100 tons of supplies.", "score": 0.79},
    {"news_id": "N095", "title": "Ocean Cleanup Project Removes 100,000 Tons of Plastic from Pacific", "category": "science", "subcategory": "environment", "abstract": "The Ocean Cleanup's expanded fleet of passive collection systems has now removed 100,000 metric tons of plastic from the Great Pacific Garbage Patch, with advanced processing converting 40% of the retrieved material into durable products.", "score": 0.77},
    {"news_id": "N096", "title": "World's First Lab-Grown Organ Successfully Transplanted in Human", "category": "science", "subcategory": "biology", "abstract": "Surgeons at Duke University Medical Center transplanted a bioengineered kidney grown from the patient's own stem cells, eliminating rejection risk and removing the need for immunosuppressant drugs. The patient, who had been on dialysis for eight years, is recovering well.", "score": 0.75},

    # ── FOOD AND DRINK ────────────────────────────────────────────────────────
    {"news_id": "N097", "title": "The Perfect Homemade Sourdough Bread Recipe That Never Fails", "category": "foodanddrink", "subcategory": "recipes", "abstract": "A professional baker shares the meticulously tested recipe and technique for reliably perfect sourdough, covering starter maintenance, hydration ratios, bulk fermentation timing and the scoring patterns that produce that signature crackling crust.", "score": 0.87},
    {"news_id": "N098", "title": "Michelin 2025: Five Surprising New Three-Star Restaurant Winners", "category": "foodanddrink", "subcategory": "restaurants", "abstract": "The Michelin Guide's 2025 edition elevated five restaurants to the coveted three-star tier, including two that had been open for less than 18 months. Most surprising was the debut of a taco stand in Mexico City and a 12-seat counter in rural Japan.", "score": 0.85},
    {"news_id": "N099", "title": "Gut Health Revolution: How Your Microbiome Controls Everything", "category": "foodanddrink", "subcategory": "nutrition", "abstract": "Gastroenterologists and microbiome researchers explain the latest science linking gut bacteria to mood, immune function, weight management and disease risk. They recommend specific fermented foods, prebiotic fibers and lifestyle changes that measurably improve microbial diversity.", "score": 0.83},
    {"news_id": "N100", "title": "Best Wines Under $20 That Taste Like They Cost $100", "category": "foodanddrink", "subcategory": "wine", "abstract": "A panel of master sommeliers blind-tasted 200 bottles priced under $20 and identified 15 that consistently fooled them as expensive wines. The list is dominated by Spanish reds, Chilean whites and Oregon Pinot Gris.", "score": 0.81},
    {"news_id": "N101", "title": "Professional Chefs Share Their Non-Negotiable Kitchen Equipment", "category": "foodanddrink", "subcategory": "cooking", "abstract": "Award-winning chefs reveal the specific knives, pans, thermometers and small appliances they consider essential for producing restaurant-quality results at home. The list is shorter than expected and focused entirely on quality over quantity.", "score": 0.79},
    {"news_id": "N102", "title": "30-Minute Meals That Actually Taste Like They Took All Day", "category": "foodanddrink", "subcategory": "recipes", "abstract": "Culinary experts share the preparation shortcuts, pantry essentials and cooking techniques that compress hours of cooking into 30 minutes without sacrificing flavor. The key is building complexity through high-heat caramelization and quality canned ingredients.", "score": 0.77},
    {"news_id": "N103", "title": "Top 10 New Restaurant Openings Worth Making a Reservation For in 2025", "category": "foodanddrink", "subcategory": "restaurants", "abstract": "Food critics from major publications have pooled their picks for the most exciting new restaurant openings of 2025, spanning formats from zero-waste tasting menus to elevated fast food concepts reimagining American classics.", "score": 0.75},
    {"news_id": "N104", "title": "Plant-Based Proteins Ranked by Nutrition, Taste and Versatility", "category": "foodanddrink", "subcategory": "nutrition", "abstract": "Registered dietitians rank the most popular plant-based protein sources by amino acid completeness, digestibility, culinary versatility and taste. Edamame, tempeh and lentils top the list while newer engineered proteins divide opinion.", "score": 0.73},
    {"news_id": "N105", "title": "Natural Wine Explained: Why Sommeliers Can't Stop Talking About It", "category": "foodanddrink", "subcategory": "wine", "abstract": "A master of wine demystifies the natural wine movement, explaining what it means to produce wine with minimal intervention, how to identify genuine natural producers versus marketing copycats, and which bottles best introduce newcomers to the category.", "score": 0.71},
    {"news_id": "N106", "title": "Mastering French Cooking Techniques at Home Without Culinary School", "category": "foodanddrink", "subcategory": "cooking", "abstract": "A Michelin-trained chef breaks down the five foundational French techniques — proper stock-making, sauce emulsification, precise knife skills, temperature control and mise en place — that elevate every dish cooked with them.", "score": 0.69},
    {"news_id": "N107", "title": "Traditional Holiday Recipes From 20 Countries Around the World", "category": "foodanddrink", "subcategory": "recipes", "abstract": "Food anthropologists and home cooks share the traditional dishes central to celebrations in 20 countries, from Mexico's ponche punch and Spain's turrón to Ethiopia's injera feast and Japan's osechi ryōri, with adaptable recipes for the modern kitchen.", "score": 0.67},
    {"news_id": "N108", "title": "Food Combining: Which Foods to Eat Together for Optimal Digestion", "category": "foodanddrink", "subcategory": "nutrition", "abstract": "Gastroenterologists separate food combining fact from fiction, identifying the combinations with real scientific evidence for digestive benefits and debunking myths that have no empirical support. High-fiber combinations with probiotic foods lead the evidence.", "score": 0.65},

    # ── LIFESTYLE ────────────────────────────────────────────────────────────
    {"news_id": "N109", "title": "Spring Fashion Trends 2025: The Colors and Silhouettes Dominating Runways", "category": "lifestyle", "subcategory": "fashion", "abstract": "Fashion directors at Vogue, Harper's Bazaar and Elle summarize the key trends emerging from the spring runway shows: oversized linen suiting, terracotta and sage palettes, updated ballet flats and a return of visible craft and handwork in ready-to-wear.", "score": 0.86},
    {"news_id": "N110", "title": "Minimalist Home Design Trends Reshaping Interiors in 2025", "category": "lifestyle", "subcategory": "home", "abstract": "Interior designers explain the shift toward intentional minimalism characterized by warm natural materials, concealed storage and a 'buy less, buy better' philosophy. The trend reflects a broader consumer push against fast furniture and disposable decor.", "score": 0.84},
    {"news_id": "N111", "title": "What Relationship Experts Say Are Signs You've Found Your Partner", "category": "lifestyle", "subcategory": "relationships", "abstract": "Marriage counselors and attachment theorists share the empirically validated indicators of a healthy long-term partnership, moving beyond romantic chemistry to focus on communication patterns, conflict resolution styles and shared values.", "score": 0.82},
    {"news_id": "N112", "title": "Morning Routine Habits Shared by the World's Most Productive People", "category": "lifestyle", "subcategory": "wellness", "abstract": "A study of 200 high-achieving executives, athletes and artists found consistent morning routines anchored by three practices: no-screen time in the first hour, physical movement before 8am, and a structured prioritization ritual before checking communications.", "score": 0.80},
    {"news_id": "N113", "title": "Skincare Ingredients That Actually Work, According to Dermatologists", "category": "lifestyle", "subcategory": "beauty", "abstract": "Board-certified dermatologists cut through skincare marketing to identify the small number of active ingredients with strong clinical evidence: retinoids, niacinamide, vitamin C, sunscreen and hydroxy acids. Everything else, they say, is largely cosmetic.", "score": 0.78},
    {"news_id": "N114", "title": "Sustainable Fashion Brands That Are Actually Changing the Industry", "category": "lifestyle", "subcategory": "fashion", "abstract": "Environmental journalists identify the fashion brands with verifiable supply chain transparency, living wages for garment workers and circular design principles, distinguishing genuine leaders from brands engaged in greenwashing.", "score": 0.76},
    {"news_id": "N115", "title": "Smart Home Devices That Actually Make Life Easier in 2025", "category": "lifestyle", "subcategory": "home", "abstract": "Consumer technology reviewers test 50 smart home products to identify the handful that deliver genuine convenience rather than gadget complexity. Smart thermostats, voice-controlled lighting and robot vacuums consistently top the practical value rankings.", "score": 0.74},
    {"news_id": "N116", "title": "Long-Distance Relationships: What Science Says Actually Keeps Them Strong", "category": "lifestyle", "subcategory": "relationships", "abstract": "Relationship psychologists share research findings on what distinguishes long-distance couples who thrive from those who struggle: structured communication rituals, clear reunion timelines and treating visits as relationship investments rather than pressure-filled events.", "score": 0.72},
    {"news_id": "N117", "title": "Sleep Science Explains Why 7-9 Hours Is the Magic Number for Adults", "category": "lifestyle", "subcategory": "wellness", "abstract": "Sleep researchers at the National Sleep Foundation explain the physiology behind the 7-9 hour recommendation, detailing what happens during REM and slow-wave sleep stages and why chronic undersleeping accelerates aging and impairs cognitive function.", "score": 0.70},
    {"news_id": "N118", "title": "The No-Makeup Makeup Look: Exact Products and Application Techniques", "category": "lifestyle", "subcategory": "beauty", "abstract": "Professional makeup artists break down the products and application order that create a perfected natural appearance, covering skin prep, strategic concealer placement, brow grooming and lip products that enhance without adding noticeable color.", "score": 0.68},
    {"news_id": "N119", "title": "How to Shop Thrift Stores for Vintage Fashion Like a Professional", "category": "lifestyle", "subcategory": "fashion", "abstract": "Vintage fashion dealers share their systematic approach to thrift store shopping: knowing which days inventory rotates, how to assess garment quality quickly, identify valuable labels and what alterations can transform cheap finds into standout pieces.", "score": 0.66},
    {"news_id": "N120", "title": "Home Office Setups That Measurably Boost Focus and Productivity", "category": "lifestyle", "subcategory": "home", "abstract": "Ergonomics consultants and productivity researchers identify the desk height, monitor distance, ambient lighting temperature and sound management strategies that have the strongest evidence for reducing fatigue and improving sustained concentration during remote work.", "score": 0.64},

    # ── AUTOS ────────────────────────────────────────────────────────────────
    {"news_id": "N121", "title": "Tesla Model 3 Highland Review: What's New and Is It Worth Buying?", "category": "autos", "subcategory": "electricvehicles", "abstract": "Auto journalists conduct a comprehensive review of the refreshed Tesla Model 3, evaluating improvements to its interior quality, updated autopilot hardware, longer range battery and revised suspension tuning against rivals from BMW and Hyundai.", "score": 0.89},
    {"news_id": "N122", "title": "Ford F-150 Lightning vs Rivian R1T: Which Electric Truck Wins in 2025?", "category": "autos", "subcategory": "electricvehicles", "abstract": "Consumer Reports tests both electric pickup trucks across towing capacity, real-world range, charging speeds and off-road capability. The Rivian edges out the Ford on software refinement while the F-150 Lightning wins on service network and familiarity.", "score": 0.87},
    {"news_id": "N123", "title": "New BMW M3 Competition xDrive Review: The Benchmark Redefined", "category": "autos", "subcategory": "reviews", "abstract": "Car and Driver's full review of the updated BMW M3 finds a performance sedan that remains the absolute benchmark in its class, with an inline-six delivering 523 horsepower and a chassis that communicates brilliantly through every corner.", "score": 0.85},
    {"news_id": "N124", "title": "Formula 1 2025 Season Preview: New Teams, New Rules, New Rivalries", "category": "autos", "subcategory": "racing", "abstract": "A comprehensive preview of the 2025 Formula 1 season covers the dramatic team reshuffles, revised technical regulations that promise closer racing, and the emerging rivalry between Max Verstappen and the new generation of championship contenders.", "score": 0.83},
    {"news_id": "N125", "title": "10 Most Reliable Cars of 2025 According to Consumer Reports", "category": "autos", "subcategory": "cars", "abstract": "Consumer Reports' annual reliability survey reveals the ten most dependable models based on owner surveys covering 17 problem areas. Japanese brands Toyota and Honda dominate the list while Tesla improves and several European luxury brands disappoint.", "score": 0.81},
    {"news_id": "N126", "title": "Real-World EV Range Test: How Far 15 Electric Cars Actually Drive", "category": "autos", "subcategory": "electricvehicles", "abstract": "Automotive engineers conducted standardized real-world range testing on 15 electric vehicles across highway, city and mixed driving cycles in both hot and cold weather. Results show EPA estimates are typically optimistic by 15-25% in real conditions.", "score": 0.79},
    {"news_id": "N127", "title": "Toyota Land Cruiser 2025 First Drive: An Icon Fully Reborn", "category": "autos", "subcategory": "reviews", "abstract": "Toyota has comprehensively redesigned the Land Cruiser while staying faithful to its off-road heritage. Our first drive found a supremely capable SUV with a refined twin-turbocharged V6, terrain management systems that border on the miraculous and build quality to match.", "score": 0.77},
    {"news_id": "N128", "title": "NASCAR Introduces Hybrid Powertrain for 2026 in Historic Shift", "category": "autos", "subcategory": "racing", "abstract": "NASCAR announced that its Cup Series cars will transition to hybrid powertrains for the 2026 season, incorporating an electric motor integrated with the existing V8 in a system developed with manufacturers to both boost performance and reduce fuel consumption.", "score": 0.75},
    {"news_id": "N129", "title": "Used Car Market Returns to Normal Prices After Three Volatile Years", "category": "autos", "subcategory": "cars", "abstract": "After three years of pandemic-driven price inflation, the used car market has finally returned to near pre-pandemic pricing levels. The adjustment has been particularly sharp for electric vehicles, creating bargain opportunities for savvy buyers.", "score": 0.73},
    {"news_id": "N130", "title": "Home EV Charging Complete Guide: Everything You Need to Know", "category": "autos", "subcategory": "electricvehicles", "abstract": "Electrical engineers and EV owners walk through the decision between Level 1 and Level 2 home charging, the installation process, recommended units, costs and the smart charging features that can substantially reduce electricity bills for EV owners.", "score": 0.71},
    {"news_id": "N131", "title": "Honda Civic Type R Is Still the Best Hot Hatch Money Can Buy", "category": "autos", "subcategory": "reviews", "abstract": "Road test editors unanimously select the Honda Civic Type R as the definitive front-wheel-drive hot hatch, praising its 315-horsepower turbocharged engine, dual-axis strut suspension and the kind of driver engagement that makes every journey an event.", "score": 0.69},
    {"news_id": "N132", "title": "Classic Car Investment Guide: Which Vintage Models Are Appreciating", "category": "autos", "subcategory": "cars", "abstract": "Hagerty auction house experts analyze the vintage car market to identify the models most likely to appreciate over the next decade, covering air-cooled Porsches, American muscle, Japanese classics and European sports cars with growing collector demand.", "score": 0.67},

    # ── WEATHER ─────────────────────────────────────────────────────────────
    {"news_id": "N133", "title": "2025 Hurricane Season Expected to Be Most Active in Recorded History", "category": "weather", "subcategory": "storms", "abstract": "NOAA's updated seasonal forecast calls for 27 named storms, 15 hurricanes and 8 major hurricanes, making 2025 potentially the most active Atlantic hurricane season ever documented. Record warm ocean temperatures are cited as the primary driver.", "score": 0.92},
    {"news_id": "N134", "title": "El Niño Returns: Global Weather Pattern Disruptions Explained", "category": "weather", "subcategory": "climate", "abstract": "Climate scientists confirm the return of a strong El Niño event that is expected to shift precipitation patterns globally, bringing drought to Australia and Southeast Asia while increasing rainfall in South America and causing warmer, drier conditions across the US south.", "score": 0.90},
    {"news_id": "N135", "title": "Winter Storm Warning: Heavy Snowfall Forecast Across Northeast US", "category": "weather", "subcategory": "forecast", "abstract": "The National Weather Service has issued winter storm warnings for 11 northeastern states as a powerful nor'easter approaches. Forecasters predict 18-24 inches of snow in major cities, with coastal areas facing blizzard conditions and significant icing.", "score": 0.88},
    {"news_id": "N136", "title": "Tornado Alley Is Expanding: New Climate Patterns Threaten More States", "category": "weather", "subcategory": "storms", "abstract": "Meteorological research shows that Tornado Alley has shifted and expanded eastward into Tennessee, Kentucky and parts of the Southeast, bringing severe tornado risk to densely populated communities with fewer storm shelters and less tornado preparedness culture.", "score": 0.86},
    {"news_id": "N137", "title": "Summer 2025 Forecast: Record Heat Expected Across Most of the US", "category": "weather", "subcategory": "forecast", "abstract": "Climate scientists warn that summer 2025 is projected to be the hottest on record for most of the continental United States, with 30+ days above 100°F expected in the Southwest and heat indices reaching dangerous levels in the humid Southeast.", "score": 0.84},
    {"news_id": "N138", "title": "Extreme Weather Frequency Increasing Faster Than Models Predicted", "category": "weather", "subcategory": "climate", "abstract": "A comprehensive analysis of 40 years of weather data confirms that the frequency and intensity of extreme weather events including floods, droughts, heatwaves and wildfires has increased 60% faster than the most aggressive climate model projections from 2000.", "score": 0.82},
    {"news_id": "N139", "title": "Spring Flood Risk Elevated Across Midwest as Snowpack Breaks Records", "category": "weather", "subcategory": "forecast", "abstract": "The US Army Corps of Engineers is monitoring record snowpack levels across the upper Midwest that, combined with saturated soils and early spring warmth, create conditions for major flooding along the Missouri and upper Mississippi river basins.", "score": 0.80},
    {"news_id": "N140", "title": "Wildfire Season Starting Weeks Earlier Every Year Due to Climate Shift", "category": "weather", "subcategory": "storms", "abstract": "Data from the US Forest Service confirms that the average wildfire season start date has moved 27 days earlier over the past 40 years across the western United States. Drought, heat and an expanding population in the wildland-urban interface are compounding the risk.", "score": 0.78},
    {"news_id": "N141", "title": "How to Winterize Your Home Against Extreme Cold Weather Events", "category": "weather", "subcategory": "forecast", "abstract": "Home improvement experts and energy engineers walk through the specific insulation, pipe protection, backup heating and emergency preparation steps homeowners should take before winter, prioritizing the highest-impact and lowest-cost measures.", "score": 0.76},
    {"news_id": "N142", "title": "Ocean Temperatures Reach Record High, Threatening Global Marine Life", "category": "weather", "subcategory": "climate", "abstract": "Global average ocean surface temperatures have reached unprecedented levels for the third consecutive year, triggering mass coral bleaching events on 80% of the world's reefs and disrupting fish migration patterns that millions of people depend on for food.", "score": 0.74},
    {"news_id": "N143", "title": "Lightning Safety: What to Do and Where to Shelter During Thunderstorms", "category": "weather", "subcategory": "storms", "abstract": "The National Weather Service updates its lightning safety guidelines with new research on strike distance, the '30-30 rule' for assessing danger, safe shelter types and the specific outdoor activities that carry the greatest risk.", "score": 0.72},
    {"news_id": "N144", "title": "Pollen Season 2025: Worst Allergies in Decades, Experts Warn", "category": "weather", "subcategory": "forecast", "abstract": "Allergists and botanists warn that 2025 will likely bring the most severe pollen season in decades, driven by an especially mild winter that failed to suppress tree populations and climate-driven extension of the pollen season at both ends.", "score": 0.70},
]

# Build lookup indices
_ARTICLE_BY_ID: dict[str, dict] = {a["news_id"]: a for a in MOCK_ARTICLES}
_BY_CATEGORY: dict[str, list[dict]] = {}
for _art in MOCK_ARTICLES:
    _BY_CATEGORY.setdefault(_art["category"], []).append(_art)
ALL_CATEGORIES: list[str] = sorted(_BY_CATEGORY.keys())

# ---------------------------------------------------------------------------
# Mock user profiles (demo accounts)
# ---------------------------------------------------------------------------

MOCK_USER_PROFILES: dict[str, dict] = {
    "U1001": {
        "display_name": "Alex — Sports Fan",
        "top_categories": ["sports"],
        "history": ["N001", "N002", "N003", "N004", "N005", "N006", "N007", "N008", "N009", "N010", "N011", "N012"],
    },
    "U1002": {
        "display_name": "Jordan — Tech Enthusiast",
        "top_categories": ["technology", "science"],
        "history": ["N025", "N026", "N027", "N028", "N029", "N030", "N031", "N032", "N033", "N034", "N085", "N088"],
    },
    "U1003": {
        "display_name": "Morgan — Health & Wellness",
        "top_categories": ["health", "lifestyle"],
        "history": ["N013", "N014", "N015", "N016", "N017", "N018", "N019", "N020", "N109", "N112", "N113", "N117"],
    },
    "U1004": {
        "display_name": "Riley — News & Politics",
        "top_categories": ["politics", "finance"],
        "history": ["N037", "N038", "N039", "N040", "N041", "N042", "N043", "N049", "N050", "N055"],
    },
    "U1005": {
        "display_name": "Sam — Finance & Markets",
        "top_categories": ["finance", "technology"],
        "history": ["N049", "N050", "N051", "N052", "N053", "N054", "N055", "N056", "N057", "N058", "N059"],
    },
    "U1006": {
        "display_name": "Taylor — Curious Reader",
        "top_categories": ["science", "travel", "entertainment"],
        "history": ["N061", "N062", "N073", "N074", "N085", "N086", "N087", "N090", "N097", "N121"],
    },
}


# ---------------------------------------------------------------------------
# Metric helpers (pure functions, work for both real and mock mode)
# ---------------------------------------------------------------------------

def _gini(categories: list[str]) -> float:
    if not categories:
        return 0.0
    counts = list(Counter(categories).values())
    n = len(counts)
    if n == 1:
        return 1.0
    counts_sorted = sorted(counts)
    cum = 0.0
    for i, c in enumerate(counts_sorted):
        cum += (2 * (i + 1) - n - 1) * c
    total = sum(counts)
    return abs(cum) / (n * total) if total else 0.0


def _ild(rec_ids: list[str], article_db: dict[str, dict]) -> float:
    """Category-based intra-list diversity (0 = all same category, 1 = all different)."""
    cats = [article_db[i]["category"] for i in rec_ids if i in article_db]
    if len(cats) < 2:
        return 0.0
    pairs = len(cats) * (len(cats) - 1) / 2
    diff = sum(1 for i in range(len(cats)) for j in range(i + 1, len(cats)) if cats[i] != cats[j])
    return diff / pairs if pairs else 0.0


def _entropy(categories: list[str]) -> float:
    if not categories:
        return 0.0
    counts = Counter(categories)
    n = len(categories)
    return -sum((c / n) * math.log2(c / n) for c in counts.values())


def _coverage(categories: list[str], all_cats: list[str]) -> float:
    return len(set(categories)) / len(all_cats) if all_cats else 0.0


# ---------------------------------------------------------------------------
# Mock recommendation helpers
# ---------------------------------------------------------------------------

def _exclude(articles: list[dict], history: list[str]) -> list[dict]:
    hist_set = set(history)
    return [a for a in articles if a["news_id"] not in hist_set]


def _mock_baseline(history: list[str], k: int) -> list[dict]:
    """Return top-k popular articles not in history."""
    candidates = _exclude(MOCK_ARTICLES, history)
    return sorted(candidates, key=lambda a: a["score"], reverse=True)[:k]


def _mock_mmr(history: list[str], k: int, lambda_param: float) -> list[dict]:
    """MMR: balance relevance vs intra-list diversity by category."""
    candidates = _exclude(MOCK_ARTICLES, history)
    if not candidates:
        return []
    selected: list[dict] = []
    remaining = list(candidates)
    selected_cats: list[str] = []
    while len(selected) < k and remaining:
        best = max(
            remaining,
            key=lambda a: (
                lambda_param * a["score"]
                - (1 - lambda_param) * (1 if a["category"] in selected_cats else 0)
            ),
        )
        selected.append(best)
        selected_cats.append(best["category"])
        remaining.remove(best)
    return selected


def _mock_calibrated(history: list[str], k: int, alpha: float) -> list[dict]:
    """Calibrated: match category distribution of history."""
    candidates = _exclude(MOCK_ARTICLES, history)
    if not candidates:
        return []
    if history:
        hist_cats = [_ARTICLE_BY_ID[h]["category"] for h in history if h in _ARTICLE_BY_ID]
        hist_dist = Counter(hist_cats)
        total = sum(hist_dist.values()) or 1
        hist_frac = {c: v / total for c, v in hist_dist.items()}
    else:
        hist_frac = {c: 1 / len(ALL_CATEGORIES) for c in ALL_CATEGORIES}

    selected: list[dict] = []
    remaining = list(candidates)
    selected_cats: list[str] = []
    while len(selected) < k and remaining:
        target_frac = hist_frac.get
        sel_total = len(selected_cats) + 1

        def score_fn(a: dict) -> float:
            cat_count = selected_cats.count(a["category"]) + 1
            cur_frac = cat_count / sel_total
            target = target_frac(a["category"], 1 / len(ALL_CATEGORIES))
            cal_bonus = max(0.0, target - cur_frac)
            return (1 - alpha) * a["score"] + alpha * cal_bonus

        best = max(remaining, key=score_fn)
        selected.append(best)
        selected_cats.append(best["category"])
        remaining.remove(best)
    return selected


def _mock_serendipity(history: list[str], k: int, beta: float) -> list[dict]:
    """Serendipity: prefer articles from categories NOT in user history."""
    candidates = _exclude(MOCK_ARTICLES, history)
    if not candidates:
        return []
    hist_cats = {_ARTICLE_BY_ID[h]["category"] for h in history if h in _ARTICLE_BY_ID}

    def score_fn(a: dict) -> float:
        serendipity_bonus = beta if a["category"] not in hist_cats else 0.0
        return (1 - beta) * a["score"] + serendipity_bonus

    return sorted(candidates, key=score_fn, reverse=True)[:k]


def _mock_composite(history: list[str], k: int, **params: Any) -> list[dict]:
    """Composite: all 4 diversity dimensions active simultaneously with per-weight control."""
    w_rel  = params.get("w_relevance",   0.40)
    w_div  = params.get("w_diversity",   0.15)
    w_cal  = params.get("w_calibration", 0.15)
    w_ser  = params.get("w_serendipity", 0.15)
    w_fair = params.get("w_fairness",    0.15)
    explore = params.get("explore_weight", 0.30)

    candidates = _exclude(MOCK_ARTICLES, history)
    if not candidates:
        return []

    hist_cats_set  = {_ARTICLE_BY_ID[h]["category"] for h in history if h in _ARTICLE_BY_ID}
    hist_cat_list  = [_ARTICLE_BY_ID[h]["category"] for h in history if h in _ARTICLE_BY_ID]
    hist_scores    = [_ARTICLE_BY_ID[h]["score"]    for h in history if h in _ARTICLE_BY_ID]
    user_pop_mean  = sum(hist_scores) / len(hist_scores) if hist_scores else 0.5

    if hist_cat_list:
        cat_counts = Counter(hist_cat_list)
        total = sum(cat_counts.values())
        hist_dist = {c: v / total for c, v in cat_counts.items()}
    else:
        hist_dist = {}

    uniform = 1.0 / max(len(ALL_CATEGORIES), 1)
    target_dist = {
        c: (1 - explore) * hist_dist.get(c, 0.0) + explore * uniform
        for c in ALL_CATEGORIES
    }

    selected: list[dict] = []
    remaining = list(candidates)
    selected_cats: list[str] = []

    while len(selected) < k and remaining:
        n_sel = max(1, len(selected))
        sc_snap = list(selected_cats)

        def score_fn(a: dict, _n: int = n_sel, _sc: list = sc_snap) -> float:
            cat = a["category"]
            # 1. Relevance
            rel = a["score"]
            # 2. Diversity: penalise repeated categories
            div_score = 1.0 - _sc.count(cat) / _n
            # 3. Calibration: reward filling under-represented categories
            cur_prop = _sc.count(cat) / _n
            tgt_prop = target_dist.get(cat, uniform)
            cal_score = min(1.0, max(0.0, (tgt_prop - cur_prop) * len(ALL_CATEGORIES)))
            # 4. Serendipity: prefer categories outside reading history
            ser_score = 0.0 if cat in hist_cats_set else 1.0
            # 5. Fairness: prefer articles whose popularity matches user preference
            fair_score = 1.0 - abs(a["score"] - user_pop_mean)
            return (
                w_rel  * rel
                + w_div  * div_score
                + w_cal  * cal_score
                + w_ser  * ser_score
                + w_fair * fair_score
            )

        best = max(remaining, key=score_fn)
        selected.append(best)
        selected_cats.append(best["category"])
        remaining.remove(best)

    return selected


def _mock_xquad(history: list[str], k: int, lambda_param: float) -> list[dict]:
    """xQuAD: proportional category coverage with diversity bonus."""
    candidates = _exclude(MOCK_ARTICLES, history)
    if not candidates:
        return []
    selected: list[dict] = []
    remaining = list(candidates)
    selected_cats: list[str] = []
    covered_cats: set[str] = set()
    n_cats = len(ALL_CATEGORIES)

    while len(selected) < k and remaining:
        def score_fn(a: dict) -> float:
            relevance = (1 - lambda_param) * a["score"]
            novelty = lambda_param * (1 if a["category"] not in covered_cats else 0.5 / (selected_cats.count(a["category"]) + 1))
            return relevance + novelty

        best = max(remaining, key=score_fn)
        selected.append(best)
        selected_cats.append(best["category"])
        covered_cats.add(best["category"])
        remaining.remove(best)
    return selected


# ---------------------------------------------------------------------------
# RecommenderService
# ---------------------------------------------------------------------------

class RecommenderService:
    """Wraps ML models with automatic fallback to mock mode.

    By default the service runs in **mock/demo mode** — 144 curated articles
    across all 12 quiz categories (including science, technology, politics).
    Set the environment variable ``NEWSLENS_REAL_MODE=1`` to load the real
    MIND-dataset models instead (note: MIND does not have science/technology/
    politics categories, so those quiz topics will fall back to 'news').
    """

    def __init__(self, base_dir: Path):
        import os
        self.mock_mode = True          # demo mode by default
        self._article_db: dict[str, dict] = dict(_ARTICLE_BY_ID)
        self._news_df = None
        self._baseline = None
        self._reranker = None
        self._analyzer = None

        use_real = os.environ.get("NEWSLENS_REAL_MODE", "").lower() in ("1", "true", "yes")
        if use_real:
            try:
                self._load_real_models(base_dir)
                self.mock_mode = False
                logger.info("RecommenderService: real MIND models loaded successfully")
            except Exception as exc:
                logger.warning(
                    "RecommenderService: real models unavailable (%s). Falling back to mock mode.", exc
                )
                self.mock_mode = True
        else:
            logger.info(
                "RecommenderService: running in mock/demo mode "
                "(set NEWSLENS_REAL_MODE=1 to use real MIND data)"
            )

    # ------------------------------------------------------------------
    # Real model loading
    # ------------------------------------------------------------------

    def _load_real_models(self, base_dir: Path) -> None:
        root = base_dir.parent.parent  # capstone_code root

        for p in [
            root / "Phase2_baseline_rec",
            root / "Phase3_echo_chambers",
            root / "Phase4_reranker",
            root / "Phase1_NLP_encoding",
            root / "Phase0_data_processing" / "data_processing",
        ]:
            if p.exists() and str(p) not in sys.path:
                sys.path.insert(0, str(p))

        from baseline_recommender_phase2 import BaselineRecommender  # noqa: PLC0415
        from diversity_reranker import DiversityReranker  # noqa: PLC0415
        from echo_chamber_analyzer import EchoChamberAnalyzer  # noqa: PLC0415
        import ast as _ast  # noqa: PLC0415
        import pandas as pd  # noqa: PLC0415

        baseline_path = str(
            root / "Phase2_baseline_rec" / "outputs" / "baseline" / "baseline_recommender.pkl"
        )
        embeddings_dir = str(root / "Phase1_NLP_encoding" / "embeddings")
        news_path = str(root / "Phase0_data_processing" / "processed_data" / "news_features_train.csv")
        behaviors_path = root / "Phase0_data_processing" / "processed_data" / "sample_train_interactions.csv"

        self._baseline = BaselineRecommender.load(baseline_path, embeddings_dir)
        self._news_df = pd.read_csv(news_path)
        news_categories = dict(zip(self._news_df["news_id"], self._news_df["category"]))

        self._reranker = DiversityReranker(
            baseline_recommender=self._baseline,
            embeddings=self._baseline.final_embeddings,
            news_id_to_idx=self._baseline.news_id_to_idx,
            news_categories=news_categories,
            popularity_scores=self._baseline.popularity_scores,
        )
        self._analyzer = EchoChamberAnalyzer(
            recommender=self._baseline,
            news_df=self._news_df,
            embeddings=self._baseline.final_embeddings,
            news_id_to_idx=self._baseline.news_id_to_idx,
        )

        # Build article DB from real news_df
        for _, row in self._news_df.iterrows():
            self._article_db[row["news_id"]] = {
                "news_id": row["news_id"],
                "title": row.get("title", ""),
                "category": row.get("category", "unknown"),
                "subcategory": row.get("subcategory", ""),
                "abstract": row.get("abstract", ""),
                "score": float(self._baseline.popularity_scores.get(row["news_id"], 0.5)),
            }

        # Load real user histories from sample_train_interactions.csv
        self._real_user_profiles: dict[str, dict] = {}
        if behaviors_path.exists():
            bdf = pd.read_csv(behaviors_path)
            for _, row in bdf.iterrows():
                uid = str(row["user_id"])
                if uid in self._real_user_profiles:
                    continue
                raw = row.get("history", "[]")
                try:
                    history = _ast.literal_eval(raw) if isinstance(raw, str) else []
                except Exception:
                    history = []
                if not history:
                    continue
                # Compute top categories from history
                hist_cats = [
                    news_categories.get(nid, "")
                    for nid in history
                    if news_categories.get(nid)
                ]
                top_cats = [c for c, _ in Counter(hist_cats).most_common(3)]
                self._real_user_profiles[uid] = {
                    "display_name": uid,
                    "history": history,
                    "top_categories": top_cats,
                }
            logger.info(
                "Loaded %d real user profiles from behaviors data",
                len(self._real_user_profiles),
            )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get_available_users(self) -> list[dict]:
        """Return list of demo user profiles for the login page."""
        if self.mock_mode:
            return [
                {
                    "user_id": uid,
                    "display_name": info["display_name"],
                    "top_categories": info["top_categories"],
                    "history_count": len(info["history"]),
                }
                for uid, info in MOCK_USER_PROFILES.items()
            ]
        # Real mode: return the actual test-set users loaded from behaviors CSV
        real = getattr(self, "_real_user_profiles", {})
        if real:
            return [
                {
                    "user_id": uid,
                    "display_name": uid,
                    "top_categories": info["top_categories"],
                    "history_count": len(info["history"]),
                }
                for uid, info in real.items()
            ]
        return []

    def get_user_info(self, user_id: str) -> dict | None:
        """Return history and metadata for a known user."""
        # Real mode: check real user profiles first
        real = getattr(self, "_real_user_profiles", {})
        if user_id in real:
            return real[user_id]
        # Fallback: mock profiles (work in both modes)
        if user_id in MOCK_USER_PROFILES:
            return MOCK_USER_PROFILES[user_id]
        return None

    def get_article(self, news_id: str) -> dict | None:
        return self._article_db.get(news_id)

    def get_popular_by_category(self, category: str, k: int = 5) -> list[dict]:
        if self.mock_mode:
            pool = _BY_CATEGORY.get(category, [])
            return sorted(pool, key=lambda a: a["score"], reverse=True)[:k]
        # Real mode: filter news_df, with alias fallback for categories not in MIND
        if self._news_df is None:
            return []
        candidates_to_try = [category] + _CATEGORY_ALIASES.get(category, [])
        for cat in candidates_to_try:
            mask = self._news_df["category"] == cat
            subset = self._news_df[mask]
            if len(subset) == 0:
                continue
            arts = []
            for _, row in subset.iterrows():
                nid = row["news_id"]
                arts.append({
                    "news_id": nid,
                    "title": row.get("title", ""),
                    "category": row.get("category", "unknown"),
                    "subcategory": row.get("subcategory", ""),
                    "abstract": row.get("abstract", ""),
                    "score": float(self._baseline.popularity_scores.get(nid, 0.0)),
                })
            arts.sort(key=lambda a: a["score"], reverse=True)
            return arts[:k]
        return []

    def recommend(self, history: list[str], k: int = 10) -> list[dict]:
        """Return top-k recommended articles (dict list)."""
        if self.mock_mode:
            return _mock_baseline(history, k)
        # Real mode
        baseline_candidates = self._baseline.recommend(
            user_history=history, k=100, exclude_history=True
        )
        results = []
        for nid, score in baseline_candidates[:k]:
            art = self._article_db.get(nid)
            if art:
                results.append({**art, "score": float(score)})
        return results

    def rerank(
        self,
        history: list[str],
        k: int = 10,
        method: str = "mmr",
        **params: Any,
    ) -> list[dict]:
        """Return k recommendations using the specified diversity method."""
        if self.mock_mode:
            return self._mock_rerank(history, k, method, params)
        # Real mode
        baseline_candidates = self._baseline.recommend(
            user_history=history, k=100, exclude_history=True
        )
        if method == "baseline":
            ranked = baseline_candidates[:k]
        else:
            ranked = self._reranker.rerank(
                candidates=baseline_candidates,
                user_history=history,
                k=k,
                method=method,
                **params,
            )
        results = []
        for nid, score in ranked:
            art = self._article_db.get(nid)
            if art:
                results.append({**art, "score": float(score)})
        return results

    def _mock_rerank(
        self, history: list[str], k: int, method: str, params: dict
    ) -> list[dict]:
        if method == "baseline":
            return _mock_baseline(history, k)
        if method == "composite":
            return _mock_composite(history, k, **params)
        if method == "mmr":
            return _mock_mmr(history, k, params.get("lambda_param", 0.5))
        if method == "calibrated":
            return _mock_calibrated(history, k, params.get("alpha", 0.6))
        if method == "serendipity":
            return _mock_serendipity(history, k, params.get("beta", 0.4))
        if method == "xquad":
            return _mock_xquad(history, k, params.get("lambda_param", 0.5))
        return _mock_baseline(history, k)

    def compute_metrics(
        self, rec_ids: list[str], history: list[str]
    ) -> dict:
        cats = [
            self._article_db[i]["category"]
            for i in rec_ids
            if i in self._article_db
        ]
        if not self.mock_mode and self._analyzer is not None:
            try:
                gini_val = float(self._analyzer.calculate_gini(cats))
                ild_val = float(self._analyzer.calculate_ild(rec_ids))
                cov_val = float(self._analyzer.calculate_coverage(cats))
                ent_val = float(self._analyzer.calculate_entropy(cats))
                all_cats = list(self._analyzer.all_categories)
                coverage_str = f"{len(set(cats))}/{len(all_cats)}"
                return {
                    "gini": gini_val,
                    "ild": ild_val,
                    "coverage": cov_val,
                    "coverage_str": coverage_str,
                    "entropy": ent_val,
                }
            except Exception:
                pass  # fall through to mock metrics
        gini_val = _gini(cats)
        ild_val = _ild(rec_ids, self._article_db)
        cov_val = _coverage(cats, ALL_CATEGORIES)
        ent_val = _entropy(cats)
        coverage_str = f"{len(set(cats))}/{len(ALL_CATEGORIES)}"
        return {
            "gini": round(gini_val, 3),
            "ild": round(ild_val, 3),
            "coverage": round(cov_val, 3),
            "coverage_str": coverage_str,
            "entropy": round(ent_val, 3),
        }
