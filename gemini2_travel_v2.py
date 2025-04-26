# Import standard and third-party libraries
import os  # For accessing environment variables
import uvicorn  # ASGI server for running FastAPI apps
import asyncio  # For asynchronous programming
import logging  # For logging messages and errors
from fastapi import FastAPI, HTTPException  # FastAPI framework and exception handling
from pydantic import BaseModel  # Data validation and settings management using Python type annotations
from typing import List, Optional  # For type hinting lists and optional values
from serpapi import GoogleSearch  # To interact with SerpAPI for Google search results
from crewai import Agent, Task, Crew, Process, LLM  # CrewAI framework for AI agent orchestration
from datetime import datetime  # For working with dates and times
from functools import lru_cache  # For caching function results

# Loading API Keys from environment variables, with default fallback values
GEMINI_API_KEY = os.getenv("GOOGLE_API_KEY", "AIzaSyA5ejJeiMmT2KjxaLjEp-4XdI-9ALyO-6s")
SERP_API_KEY = os.getenv("SERP_API_KEY", "f27a144cc55e337cbeb657249175cf45190654e61b246d4d36d61a6bbd76bc7e")

# Initializing Logger for the application
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Initialize and cache the Google Gemini LLM (Large Language Model) instance
@lru_cache(maxsize=1)
def initialize_llm():
    """Initialize and cache the LLM instance to avoid repeated initializations."""
    return LLM(
        model="gemini/gemini-2.0-flash",  # Specify the Gemini model version
        provider="google",                # Provider is Google
        api_key=GEMINI_API_KEY            # API key for authentication
    )

# Pydantic models for request and response data validation
class FlightRequest(BaseModel):
    origin: str  # Departure airport code
    destination: str  # Arrival airport code
    outbound_date: str  # Date of outbound flight
    return_date: str  # Date of return flight

class HotelRequest(BaseModel):
    location: str  # City or area for hotel search
    check_in_date: str  # Hotel check-in date
    check_out_date: str  # Hotel check-out date

class ItineraryRequest(BaseModel):
    destination: str  # Travel destination
    check_in_date: str  # Check-in date for hotel
    check_out_date: str  # Check-out date for hotel
    flights: str  # String representation of flight info
    hotels: str  # String representation of hotel info

class FlightInfo(BaseModel):
    airline: str  # Airline name
    price: str  # Flight price
    duration: str  # Total flight duration
    stops: str  # Number of stops
    departure: str  # Departure details
    arrival: str  # Arrival details
    travel_class: str  # Travel class (e.g., Economy)
    return_date: str  # Return date
    airline_logo: str  # URL to airline logo

class HotelInfo(BaseModel):
    name: str  # Hotel name
    price: str  # Price per night
    rating: float  # Hotel rating
    location: str  # Hotel location
    link: str  # Link to hotel details

class AIResponse(BaseModel):
    flights: List[FlightInfo] = []  # List of flight options
    hotels: List[HotelInfo] = []  # List of hotel options
    ai_flight_recommendation: str = ""  # AI's flight recommendation
    ai_hotel_recommendation: str = ""  # AI's hotel recommendation
    itinerary: str = ""  # Generated itinerary

# Initialize FastAPI application
app = FastAPI(title="Travel Planning API", version="1.1.0")

# Asynchronous function to run SerpAPI searches in a thread pool
async def run_search(params):
    """Generic function to run SerpAPI searches asynchronously."""
    try:
        return await asyncio.to_thread(lambda: GoogleSearch(params).get_dict())  # Run blocking call in thread
    except Exception as e:
        logger.exception(f"SerpAPI search error: {str(e)}")  # Log exceptions
        raise HTTPException(status_code=500, detail=f"Search API error: {str(e)}")  # Raise HTTP error

# Asynchronous function to search for flights using SerpAPI
async def search_flights(flight_request: FlightRequest):
    """Fetch real-time flight details from Google Flights using SerpAPI."""
    logger.info(f"Searching flights: {flight_request.origin} to {flight_request.destination}")  # Log search

    params = {
        "api_key": SERP_API_KEY,
        "engine": "google_flights",
        "hl": "en",
        "gl": "us",
        "departure_id": flight_request.origin.strip().upper(),  # Clean and format airport code
        "arrival_id": flight_request.destination.strip().upper(),
        "outbound_date": flight_request.outbound_date,
        "return_date": flight_request.return_date,
        "currency": "USD"
    }

    search_results = await run_search(params)  # Run the search
    if "error" in search_results:
        logger.error(f"Flight search error: {search_results['error']}")  # Log error
        return {"error": search_results["error"]}  # Return error dict

    best_flights = search_results.get("best_flights", [])  # Extract best flights
    if not best_flights:
        logger.warning("No flights found in search results")  # Log warning
        return []  # Return empty list if no flights

    formatted_flights = []  # List to hold formatted flight info
    for flight in best_flights:
        if not flight.get("flights") or len(flight["flights"]) == 0:
            continue  # Skip if no flight legs

        first_leg = flight["flights"][0]  # Use the first leg for details
        formatted_flights.append(FlightInfo(
            airline=first_leg.get("airline", "Unknown Airline"),
            price=str(flight.get("price", "N/A")),
            duration=f"{flight.get('total_duration', 'N/A')} min",
            stops="Nonstop" if len(flight["flights"]) == 1 else f"{len(flight['flights']) - 1} stop(s)",
            departure=f"{first_leg.get('departure_airport', {}).get('name', 'Unknown')} ({first_leg.get('departure_airport', {}).get('id', '???')}) at {first_leg.get('departure_airport', {}).get('time', 'N/A')}",
            arrival=f"{first_leg.get('arrival_airport', {}).get('name', 'Unknown')} ({first_leg.get('arrival_airport', {}).get('id', '???')}) at {first_leg.get('arrival_airport', {}).get('time', 'N/A')}",
            travel_class=first_leg.get("travel_class", "Economy"),
            return_date=flight_request.return_date,
            airline_logo=first_leg.get("airline_logo", "")
        ))

    logger.info(f"Found {len(formatted_flights)} flights")  # Log count
    return formatted_flights  # Return formatted flights

# Asynchronous function to search for hotels using SerpAPI
async def search_hotels(hotel_request: HotelRequest):
    """Fetch hotel information from SerpAPI."""
    logger.info(f"Searching hotels for: {hotel_request.location}")  # Log search

    params = {
        "api_key": SERP_API_KEY,
        "engine": "google_hotels",
        "q": hotel_request.location,
        "hl": "en",
        "gl": "us",
        "check_in_date": hotel_request.check_in_date,
        "check_out_date": hotel_request.check_out_date,
        "currency": "USD",
        "sort_by": 3,
        "rating": 8
    }

    search_results = await run_search(params)  # Run the search

    if "error" in search_results:
        logger.error(f"Hotel search error: {search_results['error']}")  # Log error
        return {"error": search_results["error"]}  # Return error dict

    hotel_properties = search_results.get("properties", [])  # Extract hotel properties
    if not hotel_properties:
        logger.warning("No hotels found in search results")  # Log warning
        return []  # Return empty list if no hotels

    formatted_hotels = []  # List for formatted hotel info
    for hotel in hotel_properties:
        try:
            formatted_hotels.append(HotelInfo(
                name=hotel.get("name", "Unknown Hotel"),
                price=hotel.get("rate_per_night", {}).get("lowest", "N/A"),
                rating=hotel.get("overall_rating", 0.0),
                location=hotel.get("location", "N/A"),
                link=hotel.get("link", "N/A")
            ))
        except Exception as e:
            logger.warning(f"Error formatting hotel data: {str(e)}")  # Log and skip errors

    logger.info(f"Found {len(formatted_hotels)} hotels")  # Log count
    return formatted_hotels  # Return formatted hotels

# Format travel data (flights or hotels) for AI analysis
def format_travel_data(data_type, data):
    """Generic formatter for both flight and hotel data."""
    if not data:
        return f"No {data_type} available."  # Return message if no data

    if data_type == "flights":
        formatted_text = "✈️ **Available flight options**:\n\n"
        for i, flight in enumerate(data):
            formatted_text += (
                f"**Flight {i + 1}:**\n"
                f"✈️ **Airline:** {flight.airline}\n"
                f"💰 **Price:** ${flight.price}\n"
                f"⏱️ **Duration:** {flight.duration}\n"
                f"🛑 **Stops:** {flight.stops}\n"
                f"🕔 **Departure:** {flight.departure}\n"
                f"🕖 **Arrival:** {flight.arrival}\n"
                f"💺 **Class:** {flight.travel_class}\n\n"
            )
    elif data_type == "hotels":
        formatted_text = "🏨 **Available Hotel Options**:\n\n"
        for i, hotel in enumerate(data):
            formatted_text += (
                f"**Hotel {i + 1}:**\n"
                f"🏨 **Name:** {hotel.name}\n"
                f"💰 **Price:** ${hotel.price}\n"
                f"⭐ **Rating:** {hotel.rating}\n"
                f"📍 **Location:** {hotel.location}\n"
                f"🔗 **More Info:** [Link]({hotel.link})\n\n"
            )
    else:
        return "Invalid data type."  # Handle invalid types

    return formatted_text.strip()  # Remove trailing whitespace

# Unified function for AI recommendations (flights or hotels)
async def get_ai_recommendation(data_type, formatted_data):
    """Unified function for getting AI recommendations for both flights and hotels."""
    logger.info(f"Getting {data_type} analysis from AI")  # Log action
    llm_model = initialize_llm()  # Get cached LLM

    # Configure agent and task based on data type
    if data_type == "flights":
        role = "AI Flight Analyst"
        goal = "Analyze flight options and recommend the best one considering price, duration, stops, and overall convenience."
        backstory = f"AI expert that provides in-depth analysis comparing flight options based on multiple factors."
        description = """
        Recommend the best flight from the available options, based on the details provided below:
        **Reasoning for Recommendation:**
        - **💰 Price:** Provide a detailed explanation about why this flight offers the best value compared to others.
        - **⏱️ Duration:** Explain why this flight has the best duration in comparison to others.
        - **🛑 Stops:** Discuss why this flight has minimal or optimal stops.
        - **💺 Travel Class:** Describe why this flight provides the best comfort and amenities.
        Use the provided flight data as the basis for your recommendation. Be sure to justify your choice using clear reasoning for each attribute. Do not repeat the flight details in your response.
        """
    elif data_type == "hotels":
        role = "AI Hotel Analyst"
        goal = "Analyze hotel options and recommend the best one considering price, rating, location, and amenities."
        backstory = f"AI expert that provides in-depth analysis comparing hotel options based on multiple factors."
        description = """
        Based on the following analysis, generate a detailed recommendation for the best hotel. Your response should include clear reasoning based on price, rating, location, and amenities.
        **🏆 AI Hotel Recommendation**
        We recommend the best hotel based on the following analysis:
        **Reasoning for Recommendation**:
        - **💰 Price:** The recommended hotel is the best option for the price compared to others, offering the best value for the amenities and services provided.
        - **⭐ Rating:** With a higher rating compared to the alternatives, it ensures a better overall guest experience. Explain why this makes it the best choice.
        - **📍 Location:** The hotel is in a prime location, close to important attractions, making it convenient for travelers.
        - **🛋️ Amenities:** The hotel offers amenities like Wi-Fi, pool, fitness center, free breakfast, etc. Discuss how these amenities enhance the experience, making it suitable for different types of travelers.
        📝 **Reasoning Requirements**:
        - Ensure that each section clearly explains why this hotel is the best option based on the factors of price, rating, location, and amenities.
        - Compare it against the other options and explain why this one stands out.
        - Provide concise, well-structured reasoning to make the recommendation clear to the traveler.
        - Your recommendation should help a traveler make an informed decision based on multiple factors, not just one.
        """
    else:
        raise ValueError("Invalid data type for AI recommendation")  # Raise error for invalid type

    # Create the AI agent for analysis
    analyze_agent = Agent(
        role=role,
        goal=goal,
        backstory=backstory,
        llm=llm_model,
        verbose=False
    )

    # Create the AI analysis task
    analyze_task = Task(
        description=f"{description}\n\nData to analyze:\n{formatted_data}",
        agent=analyze_agent,
        expected_output=f"A structured recommendation explaining the best {data_type} choice based on the analysis of provided details."
    )

    # Create the CrewAI crew to execute the task
    analyst_crew = Crew(
        agents=[analyze_agent],
        tasks=[analyze_task],
        process=Process.sequential,
        verbose=False
    )

    try:
        # Run the CrewAI analysis in a thread pool
        crew_results = await asyncio.to_thread(analyst_crew.kickoff)

        # Handle different possible return types from CrewAI
        if hasattr(crew_results, 'outputs') and crew_results.outputs:
            return crew_results.outputs[0]
        elif hasattr(crew_results, 'get'):
            return crew_results.get(role, f"No {data_type} recommendation available.")
        else:
            return str(crew_results)
    except Exception as e:
        logger.exception(f"Error in AI {data_type} analysis: {str(e)}")  # Log error
        return f"Unable to generate {data_type} recommendation due to an error."  # Return error message

# Generate a detailed travel itinerary using AI
async def generate_itinerary(destination, flights_text, hotels_text, check_in_date, check_out_date):
    """Generate a detailed travel itinerary based on flight and hotel information."""
    try:
        # Convert string dates to datetime objects
        check_in = datetime.strptime(check_in_date, "%Y-%m-%d")
        check_out = datetime.strptime(check_out_date, "%Y-%m-%d")
        # Calculate the number of days for the trip
        days = (check_out - check_in).days

        llm_model = initialize_llm()  # Get cached LLM

        # Create the AI travel planner agent
        analyze_agent = Agent(
            role="AI Travel Planner",
            goal="Create a detailed itinerary for the user based on flight and hotel information",
            backstory="AI travel expert generating a day-by-day itinerary including flight details, hotel stays, and must-visit locations in the destination.",
            llm=llm_model,
            verbose=False
        )

        # Create the itinerary generation task
        analyze_task = Task(
            description=f"""
            Based on the following details, create a {days}-day itinerary for the user:

            **Flight Details**:
            {flights_text}

            **Hotel Details**:
            {hotels_text}

            **Destination**: {destination}

            **Travel Dates**: {check_in_date} to {check_out_date} ({days} days)

            The itinerary should include:
            - Flight arrival and departure information
            - Hotel check-in and check-out details
            - Day-by-day breakdown of activities
            - Must-visit attractions and estimated visit times
            - Restaurant recommendations for meals
            - Tips for local transportation

            📝 **Format Requirements**:
            - Use markdown formatting with clear headings (# for main headings, ## for days, ### for sections)
            - Include emojis for different types of activities (🏛️ for landmarks, 🍽️ for restaurants, etc.)
            - Use bullet points for listing activities
            - Include estimated timings for each activity
            - Format the itinerary to be visually appealing and easy to read
            """,
            agent=analyze_agent,
            expected_output="A well-structured, visually appealing itinerary in markdown format, including flight, hotel, and day-wise breakdown with emojis, headers, and bullet points."
        )

        # Create the CrewAI crew to execute the itinerary task
        itinerary_planner_crew = Crew(
            agents=[analyze_agent],
            tasks=[analyze_task],
            process=Process.sequential,
            verbose=False
        )

        # Run the itinerary generation in a thread pool
        crew_results = await asyncio.to_thread(itinerary_planner_crew.kickoff)

        # Handle different possible return types from CrewAI
        if hasattr(crew_results, 'outputs') and crew_results.outputs:
            return crew_results.outputs[0]
        elif hasattr(crew_results, 'get'):
            return crew_results.get("AI Travel Planner", "No itinerary available.")
        else:
            return str(crew_results)

    except Exception as e:
        logger.exception(f"Error generating itinerary: {str(e)}")  # Log error
        return "Unable to generate itinerary due to an error. Please try again later."  # Return error message

# API endpoint to search for flights and get AI recommendation
@app.post("/search_flights/", response_model=AIResponse)
async def get_flight_recommendations(flight_request: FlightRequest):
    """Search flights and get AI recommendation."""
    try:
        # Search for flights
        flights = await search_flights(flight_request)

        # Handle errors
        if isinstance(flights, dict) and "error" in flights:
            raise HTTPException(status_code=400, detail=flights["error"])

        if not flights:
            raise HTTPException(status_code=404, detail="No flights found")

        # Format flight data for AI
        flights_text = format_travel_data("flights", flights)

        # Get AI recommendation
        ai_recommendation = await get_ai_recommendation("flights", flights_text)

        # Return response with flights and recommendation
        return AIResponse(
            flights=flights,
            ai_flight_recommendation=ai_recommendation
        )
    except HTTPException:
        # Re-raise HTTP exceptions to preserve status codes
        raise
    except Exception as e:
        logger.exception(f"Flight search endpoint error: {str(e)}")  # Log error
        raise HTTPException(status_code=500, detail=f"Flight search error: {str(e)}")  # Return HTTP error

# API endpoint to search for hotels and get AI recommendation
@app.post("/search_hotels/", response_model=AIResponse)
async def get_hotel_recommendations(hotel_request: HotelRequest):
    """Search hotels and get AI recommendation."""
    try:
        # Fetch hotel data
        hotels = await search_hotels(hotel_request)

        # Handle errors
        if isinstance(hotels, dict) and "error" in hotels:
            raise HTTPException(status_code=400, detail=hotels["error"])

        if not hotels:
            raise HTTPException(status_code=404, detail="No hotels found")

        # Format hotel data for AI
        hotels_text = format_travel_data("hotels", hotels)

        # Get AI recommendation
        ai_recommendation = await get_ai_recommendation("hotels", hotels_text)

        # Return response with hotels and recommendation
        return AIResponse(
            hotels=hotels,
            ai_hotel_recommendation=ai_recommendation
        )
    except HTTPException:
        # Re-raise HTTP exceptions to preserve status codes
        raise
    except Exception as e:
        logger.exception(f"Hotel search endpoint error: {str(e)}")  # Log error
        raise HTTPException(status_code=500, detail=f"Hotel search error: {str(e)}")  # Return HTTP error

# API endpoint to search for both flights and hotels concurrently, and get AI recommendations and itinerary
@app.post("/complete_search/", response_model=AIResponse)
async def complete_travel_search(flight_request: FlightRequest, hotel_request: Optional[HotelRequest] = None):
    """Search for flights and hotels concurrently and get AI recommendations for both."""
    try:
        # If hotel request is not provided, create one from flight request
        if hotel_request is None:
            hotel_request = HotelRequest(
                location=flight_request.destination,
                check_in_date=flight_request.outbound_date,
                check_out_date=flight_request.return_date
            )

        # Run flight and hotel searches concurrently
        flight_task = asyncio.create_task(get_flight_recommendations(flight_request))
        hotel_task = asyncio.create_task(get_hotel_recommendations(hotel_request))

        # Wait for both tasks to complete
        flight_results, hotel_results = await asyncio.gather(flight_task, hotel_task, return_exceptions=True)

        # Check for exceptions in flight results
        if isinstance(flight_results, Exception):
            logger.error(f"Flight search failed: {str(flight_results)}")
            flight_results = AIResponse(flights=[], ai_flight_recommendation="Could not retrieve flights.")

        # Check for exceptions in hotel results
        if isinstance(hotel_results, Exception):
            logger.error(f"Hotel search failed: {str(hotel_results)}")
            hotel_results = AIResponse(hotels=[], ai_hotel_recommendation="Could not retrieve hotels.")

        # Format data for itinerary generation
        flights_text = format_travel_data("flights", flight_results.flights)
        hotels_text = format_travel_data("hotels", hotel_results.hotels)

        # Generate itinerary if both searches were successful
        itinerary = ""
        if flight_results.flights and hotel_results.hotels:
            itinerary = await generate_itinerary(
                destination=flight_request.destination,
                flights_text=flights_text,
                hotels_text=hotels_text,
                check_in_date=flight_request.outbound_date,
                check_out_date=flight_request.return_date
            )

        # Combine results into a single response
        return AIResponse(
            flights=flight_results.flights,
            hotels=hotel_results.hotels,
            ai_flight_recommendation=flight_results.ai_flight_recommendation,
            ai_hotel_recommendation=hotel_results.ai_hotel_recommendation,
            itinerary=itinerary
        )
    except Exception as e:
        logger.exception(f"Complete travel search error: {str(e)}")  # Log error
        raise HTTPException(status_code=500, detail=f"Travel search error: {str(e)}")  # Return HTTP error

# API endpoint to generate an itinerary from provided flight and hotel information
@app.post("/generate_itinerary/", response_model=AIResponse)
async def get_itinerary(itinerary_request: ItineraryRequest):
    """Generate an itinerary based on provided flight and hotel information."""
    try:
        itinerary = await generate_itinerary(
            destination=itinerary_request.destination,
            flights_text=itinerary_request.flights,
            hotels_text=itinerary_request.hotels,
            check_in_date=itinerary_request.check_in_date,
            check_out_date=itinerary_request.check_out_date
        )

        return AIResponse(itinerary=itinerary)  # Return itinerary in response
    except Exception as e:
        logger.exception(f"Itinerary generation error: {str(e)}")  # Log error
        raise HTTPException(status_code=500, detail=f"Itinerary generation error: {str(e)}")  # Return HTTP error

# Entry point to run the FastAPI server
if __name__ == "__main__":
    logger.info("Starting Travel Planning API server")  # Log server start
    uvicorn.run(app, host="0.0.0.0", port=8000)  # Run the FastAPI app on port 8000
