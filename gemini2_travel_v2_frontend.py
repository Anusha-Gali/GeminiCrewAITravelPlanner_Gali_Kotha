# gemini2_travel_v2_frontend.py  ✈️ AI-Powered Travel Planner (v2.1.4)

import streamlit as st                # Import Streamlit for building the web UI
import requests                       # Import requests for HTTP API calls
from datetime import datetime, timedelta  # Import datetime utilities for date handling

# API End-points
API_BASE_URL      = "http://localhost:8000"           # Base URL for backend API
API_URL_FLIGHTS   = f"{API_BASE_URL}/search_flights/" # Endpoint for flight search
API_URL_HOTELS    = f"{API_BASE_URL}/search_hotels/"  # Endpoint for hotel search
API_URL_COMPLETE  = f"{API_BASE_URL}/complete_search/"# Endpoint for combined search

# Page config
st.set_page_config(
    page_title="✈️ AI-Powered Travel Planner", # Set browser tab title
    page_icon="✈️",                           # Set browser tab icon
    layout="wide",                            # Use wide layout for page
    initial_sidebar_state="expanded",         # Sidebar is expanded by default
)

# Helpers
def pretty_table(item: dict):
    """Display a dictionary as a markdown table in Streamlit."""
    header = "| Attribute | Value |\n|---|---|\n"
    rows   = "\n".join([f"| **{k}** | {v} |" for k, v in item.items()])
    st.markdown(header + rows)

def safe_rerun():
    """Trigger a rerun regardless of Streamlit version."""
    if hasattr(st, "rerun"):
        st.rerun()
    elif hasattr(st, "experimental_rerun"):
        st.experimental_rerun()

# Sidebar
with st.sidebar:
    st.title("⚙️ Options")  # Sidebar title
    search_mode = st.radio(
        "Search Mode",      # Radio button label
        ["Complete (Flights + Hotels + Itinerary)", "Flights Only", "Hotels Only"] # Modes
    )
    st.markdown("---")      # Divider line
    st.caption("AI-Powered Travel Planner v2.1.4")   # Version info
    st.caption("© 2025 Travel AI Solutions")         # Copyright

# Header
st.title("✈️ AI-Powered Travel Planner") # Main page title
st.markdown(
    "**Find flights, hotels, and get personalized AI recommendations — "
    "create your perfect itinerary in seconds.**"
) # Intro text

# Search Form
with st.form("travel_form"):
    c1, c2 = st.columns(2)  # Create two columns for form layout

    with c1:
        st.subheader("🛫 Flight Details")  # Flight details section
        origin      = st.text_input("Departure Airport (IATA)", "ATL")    # Input for origin airport
        destination = st.text_input("Arrival Airport (IATA)",  "LAX")    # Input for destination airport
        tomorrow    = datetime.now() + timedelta(days=1)                  # Default: tomorrow
        next_week   = tomorrow + timedelta(days=7)                        # Default: a week after tomorrow
        outbound_dt = st.date_input("Departure Date", tomorrow)           # Departure date input
        return_dt   = st.date_input("Return Date",   next_week)           # Return date input

    with c2:
        st.subheader("🏨 Hotel Details")                                   # Hotel details section
        use_dest  = st.checkbox("Use flight destination for hotel", True) # Checkbox: use flight destination as hotel location
        location  = destination if use_dest else st.text_input("Hotel Location", "") # Hotel location input
        check_in  = st.date_input("Check-In Date",  outbound_dt)          # Hotel check-in date
        check_out = st.date_input("Check-Out Date", return_dt)            # Hotel check-out date

    submitted = st.form_submit_button("🔍 Search", use_container_width=True) # Submit button

# Perform search
if submitted:
    # Validate user inputs
    if not origin or not destination:
        st.error("Please enter both origin and destination airports."); st.stop()
    if outbound_dt >= return_dt:
        st.error("Return date must be after departure date.");          st.stop()
    if check_in   >= check_out:
        st.error("Check-out date must be after check-in date.");        st.stop()

    # Prepare API payloads for the backend
    flight_req = {
        "origin": origin, "destination": destination,
        "outbound_date": str(outbound_dt), "return_date": str(return_dt)
    }
    hotel_req  = {
        "location": location,
        "check_in_date":  str(check_in),
        "check_out_date": str(check_out)
    }

    with st.spinner("Fetching travel options…"):
        try:
            # Call appropriate backend endpoint based on search mode
            if search_mode.startswith("Complete"):
                resp = requests.post(API_URL_COMPLETE,
                                     json={"flight_request": flight_req,
                                           "hotel_request":  hotel_req})
            elif search_mode == "Flights Only":
                resp = requests.post(API_URL_FLIGHTS, json=flight_req)
            else:  # Hotels Only
                resp = requests.post(API_URL_HOTELS,  json=hotel_req)

            resp.raise_for_status()   # Raise error if HTTP status is not 200
            data = resp.json()        # Parse JSON response
        except Exception as e:
            st.error(f"API error: {e}"); st.stop()  # Show error and stop

    # Save search results and request details in session state for rendering
    st.session_state["search_payload"] = {
        "mode":  search_mode,
        "origin": origin,
        "destination": destination,
        "location": location,
        "outbound_dt": str(outbound_dt),
        "return_dt":  str(return_dt),
        "check_in":   str(check_in),
        "check_out":  str(check_out),
        "flights":  data.get("flights", []),
        "hotels":   data.get("hotels",  []),
        "ai_frec":  data.get("ai_flight_recommendation", ""),
        "ai_hrec":  data.get("ai_hotel_recommendation",  ""),
        "itinerary":data.get("itinerary", "")
    }

# Render results
if "search_payload" in st.session_state:
    s = st.session_state["search_payload"]  # Retrieve search results and details
    flights, hotels = s["flights"], s["hotels"]
    ai_frec, ai_hrec, itinerary = s["ai_frec"], s["ai_hrec"], s["itinerary"]
    mode = s["mode"]

    # Set tab names based on search mode
    tab_names = (["✈️ Flights","🏆 AI Rec"] if mode=="Flights Only" else
                 ["🏨 Hotels","🏆 AI Rec"] if mode=="Hotels Only"  else
                 ["✈️ Flights","🏨 Hotels","🏆 AI Recs","📅 Itinerary"])
    tabs = st.tabs(tab_names)  # Create tabs

    # ── Flights tab ────────────────────
    if mode != "Hotels Only":
        with tabs[0]:
            st.subheader(f"✈️ Flights ({s['origin']} ➔ {s['destination']})") # Flights tab header
            if flights:
                c1,c2 = st.columns(2)  # Two columns for flight cards
                for i, f in enumerate(flights):
                    with (c1 if i%2==0 else c2):  # Alternate columns for flight cards
                        with st.container(border=True):
                            # Display flight details in markdown
                            st.markdown(
                                f"### ✈️ {f['airline']} – {f['stops']} Flight\n"
                                f"- 🕒 **Departure:** {f['departure']}\n"
                                f"- 🕘 **Arrival:** {f['arrival']}\n"
                                f"- ⏱️ **Duration:** {f['duration']}\n"
                                f"- 💰 **Price:** **${f['price']}**\n"
                                f"- 💺 **Class:** {f['travel_class']}"
                            )
                            left, right = st.columns(2)  # Buttons: select and details
                            with left:
                                if st.button("🔖 Select", key=f"fsel_{i}"):
                                    st.session_state["selected_flight"] = f # Save selected flight
                                    st.success("Flight saved!")
                            with right:
                                link = (f.get("booking_link")
                                        or f"https://www.google.com/travel/flights?q={s['origin']}{s['destination']}")
                                st.link_button("🔗 Details", link) # Link to booking/details

                # Show selected flight if any
                if "selected_flight" in st.session_state:
                    st.markdown("#### ✅ Selected Flight")
                    clean = {k: v for k, v in st.session_state["selected_flight"].items()
                             if k != "airline_logo"}  # Remove logo from display
                    pretty_table(clean)
                    if st.button("❌ Clear Flight Selection"):
                        st.session_state.pop("selected_flight", None) # Remove selection
                        safe_rerun()
            else:
                st.info("No flights returned from API.") # No flights info

    # ── Hotels tab ─────────────────────
    if mode != "Flights Only":
        idx = 0 if mode=="Hotels Only" else 1  # Tab index for hotels
        with tabs[idx]:
            st.subheader(f"🏨 Hotels in {s['location']}") # Hotels tab header
            if hotels:
                c1,c2,c3 = st.columns(3)  # Three columns for hotel cards
                for i, h in enumerate(hotels):
                    with [c1,c2,c3][i%3]:  # Distribute hotels across columns
                        with st.container(border=True):
                            # Display hotel details in markdown
                            st.markdown(
                                f"### 🏨 {h['name']}\n"
                                f"- 💰 **${h['price']}** /night\n"
                                f"- ⭐ **Rating:** {h['rating']}\n"
                                f"- 📍 {h['location']}"
                            )
                            left, right = st.columns(2)  # Buttons: select and details
                            with left:
                                if st.button("🔖 Select", key=f"hsel_{i}"):
                                    st.session_state["selected_hotel"] = h # Save selected hotel
                                    st.success("Hotel saved!")
                            with right:
                                st.link_button("🔗 Details", h["link"]) # Link to hotel details

                # Show selected hotel if any
                if "selected_hotel" in st.session_state:
                    st.markdown("#### ✅ Selected Hotel")
                    pretty_table(st.session_state["selected_hotel"])
                    if st.button("❌ Clear Hotel Selection"):
                        st.session_state.pop("selected_hotel", None) # Remove selection
                        safe_rerun()
            else:
                st.info("No hotels returned from API.") # No hotels info

    # ── AI Recommendations tab ─────────
    rec_idx = 1 if mode in ["Flights Only","Hotels Only"] else 2 # Tab index for AI recs
    with tabs[rec_idx]:
        if ai_frec and mode!="Hotels Only":
            st.subheader("✈️ AI Flight Recommendation")
            st.container(border=True).text(ai_frec) # Show AI flight rec as plain text
        if ai_hrec and mode!="Flights Only":
            st.subheader("🏨 AI Hotel Recommendation")
            st.container(border=True).text(ai_hrec) # Show AI hotel rec as plain text

    # ── Itinerary tab ──────────────────
    if mode.startswith("Complete") and itinerary:
        with tabs[3]:
            st.subheader("📅 Your Itinerary")
            st.container(border=True).text(itinerary) # Show itinerary as plain text
            st.download_button(
                "📥 Download Itinerary",
                data=itinerary,
                file_name=f"itinerary_{s['destination']}_{s['outbound_dt']}.md",
                mime="text/markdown"
            ) # Download itinerary as markdown file
