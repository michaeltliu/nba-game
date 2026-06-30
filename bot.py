import sys
import time
import argparse
import httpx
import itertools
import random

# Ensure the local directory is in the import path so we can import room and game
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from room import Player
from game import NBAPlayer

# Color codes for beautiful console output
class Colors:
    HEADER = "\033[95m"
    OKBLUE = "\033[94m"
    OKCYAN = "\033[96m"
    OKGREEN = "\033[92m"
    WARNING = "\033[93m"
    FAIL = "\033[91m"
    ENDC = "\033[0m"
    BOLD = "\033[1m"
    UNDERLINE = "\033[4m"

def log_info(msg: str):
    print(f"{Colors.OKBLUE}[INFO]{Colors.ENDC} {msg}")

def log_success(msg: str):
    print(f"{Colors.OKGREEN}[SUCCESS]{Colors.ENDC} {msg}")

def log_warn(msg: str):
    print(f"{Colors.WARNING}[WARN]{Colors.ENDC} {msg}")

def log_error(msg: str):
    print(f"{Colors.FAIL}[ERROR]{Colors.ENDC} {msg}")

def format_positions(player: NBAPlayer) -> str:
    positions = []
    if player.guard: positions.append("G")
    if player.forward: positions.append("F")
    if player.center: positions.append("C")
    return "/".join(positions)

def calculate_bid(
    bot_name: str,
    members: list[Player],
    player_queue: list[NBAPlayer],
    missing_position_penalty: int,
    balance: int,
    current_team: list[NBAPlayer]
) -> int:
    """
    Computes a smart bid for the player at the front of the queue using a
    Value Over Replacement Player (VORP) style valuation.
    
    1. Computes the maximum potential team score for each candidate player.
    2. Establishes a replacement baseline (average of non-target players).
    3. Calculates each player's value over replacement.
    4. Allocates the budget across players proportional to their value.
       This ensures we bid on solid players even if they aren't in our absolute
       "dream team", preventing opponents from getting them for free.
    """
    k = 5 - len(current_team)
    if k <= 0:
        return 0
        
    if not player_queue:
        return 0
        
    current_player = player_queue[0]
    penalty_factor = 1.0 / (2.0 ** (missing_position_penalty / 4.0))
    
    # Helper to compute the score of a team using the server's Player model
    def get_team_score(team: list[NBAPlayer]) -> float:
        temp_player = Player(name="temp")
        temp_player.nba_team = team
        shortfall = temp_player.best_lineup()
        return temp_player.compute_score(shortfall, penalty_factor)
        
    # Restrict our search to the first M players in the queue to keep it fast
    M = 15
    candidates = player_queue[:min(len(player_queue), M)]
    
    # 1. For each candidate player P, compute their utility U(P).
    # U(P) is the max score we can get if we draft P next and fill the remaining
    # k-1 slots optimally from the rest of the candidates.
    utilities = {}
    for P in candidates:
        other_candidates = [x for x in candidates if x.pid != P.pid]
        target_k_minus_1 = min(k - 1, len(other_candidates))
        
        best_score = -1.0
        if target_k_minus_1 == 0:
            # We only need 1 player, so the utility is just the score of current_team + P
            best_score = get_team_score(current_team + [P])
        else:
            # Find the best combination of size k-1
            for subset in itertools.combinations(other_candidates, target_k_minus_1):
                score = get_team_score(current_team + [P] + list(subset))
                if score > best_score:
                    best_score = score
        utilities[P.pid] = best_score
        
    # 2. Sort candidates by utility descending
    sorted_candidates = sorted(candidates, key=lambda x: utilities[x.pid], reverse=True)
    
    # Our target players are the top k players
    target_players = sorted_candidates[:k]
    
    # 3. Determine the replacement utility level.
    # The replacement player is the average of the players we don't target.
    if len(candidates) > k:
        replacement_players = sorted_candidates[k:]
        u_replacement = sum(utilities[p.pid] for p in replacement_players) / len(replacement_players)
    else:
        u_replacement = 0.0
        
    # 4. Compute the Value Over Replacement V(P) for each player
    values = {p.pid: max(0.0, utilities[p.pid] - u_replacement) for p in candidates}
    
    # If the current player has no value over replacement, we bid 0
    current_player_value = values[current_player.pid]
    if current_player_value <= 0:
        return 0
        
    # 5. Allocate budget proportional to value over replacement
    sum_target_v = sum(values[p.pid] for p in target_players)
    
    if sum_target_v == 0:
        # If all target players have 0 value over replacement (e.g., all identical),
        # we distribute budget equally among target players
        is_target = any(p.pid == current_player.pid for p in target_players)
        if is_target:
            bid = max(1, balance // k)
        else:
            bid = 0
    else:
        # Bid is proportional to current player's value vs the sum of target values
        bid = balance * (current_player_value / sum_target_v)
        bid = int(round(bid))
        
    # Smart micro-optimizations:
    # 1. If we want this player (value > 0) and have budget, bid at least $1 so we don't lose to a $0 bid.
    if bid == 0 and balance > 0 and current_player_value > 0:
        bid = 1
        
    # 2. Ensure the bid never exceeds our current balance or goes below 0.
    bid = max(0, min(bid, balance))
    
    return bid

def play_bot(room_code: str, api_url: str, base_name: str):
    client = httpx.Client(base_url=api_url, timeout=10.0)
    
    # 1. Join the room
    # Handle name collisions by appending a random suffix if needed
    bot_name = base_name
    player_id = None
    
    log_info(f"Attempting to join room '{room_code}'...")
    for attempt in range(5):
        try:
            response = client.post(f"/join-room/{room_code}", json={"player_name": bot_name})
            res_data = response.json()
            if res_data.get("success"):
                player_id = res_data["player_id"]
                log_success(f"Joined room successfully as '{Colors.BOLD}{bot_name}{Colors.ENDC}'!")
                break
            elif res_data.get("failure_msg") == "NAME_TAKEN":
                bot_name = f"{base_name}_{random.randint(100, 999)}"
                log_warn(f"Name taken, retrying as '{bot_name}'...")
            else:
                log_error(f"Failed to join room: {res_data.get('failure_msg')}")
                sys.exit(1)
        except Exception as e:
            log_error(f"Connection error while joining room: {e}")
            time.sleep(2)
            
    if not player_id:
        log_error("Could not join room after 5 attempts.")
        sys.exit(1)
        
    # 2. Main Polling Loop
    bot_last_bid_round = 0
    last_printed_resolved_pid = None
    in_game = False
    waiting_message_printed = False
    
    log_info("Starting room status polling loop (every 2 seconds)...")
    
    while True:
        try:
            response = client.get(f"/rooms/{room_code}/status")
            if response.status_code != 200:
                log_error(f"API returned status code {response.status_code}. Retrying...")
                time.sleep(2)
                continue
                
            status = response.json()
            if not status.get("success"):
                log_error(f"Status API failed: {status.get('failure_msg')}")
                time.sleep(2)
                continue
                
            round_num = status["round_num"]
            members_data = status["members"]
            
            # Parse members into Player objects
            members = [Player(**m) for m in members_data]
            
            # Find our own player object
            our_player = next((m for m in members if m.name == bot_name), None)
            if not our_player:
                log_error(f"Could not find ourselves ('{bot_name}') in the room members!")
                time.sleep(2)
                continue
                
            # Parse the player queue
            player_queue = [NBAPlayer(**p) for p in status["player_queue"]]
            
            # Print previous round results if available and not yet printed
            prev_result = status.get("prev_auction_result")
            if prev_result:
                prev_pid = prev_result["nba_player"]["pid"]
                if prev_pid != last_printed_resolved_pid:
                    winner_name = prev_result["winner"] or "Nobody"
                    player_name = prev_result["nba_player"]["name"]
                    price = prev_result["price_paid"]
                    print(f"\n{Colors.OKCYAN}=== Auction Result ==={Colors.ENDC}")
                    print(f"Player: {Colors.BOLD}{player_name}{Colors.ENDC}")
                    print(f"Winner: {Colors.BOLD}{winner_name}{Colors.ENDC}")
                    print(f"Price Paid: {Colors.BOLD}${price}{Colors.ENDC}")
                    print(f"======================\n")
                    last_printed_resolved_pid = prev_pid
            
            # Game State Machine
            if round_num > 0:
                # Game is in progress!
                in_game = True
                waiting_message_printed = False
                
                # Check if it's a new round that we haven't bid on yet
                if round_num > bot_last_bid_round:
                    # Check if we are expected/allowed to bid
                    # We can bid if we have < 5 players and our balance > 0
                    roster_size = len(our_player.nba_team)
                    can_bid = roster_size < 5 and our_player.balance > 0
                    
                    if can_bid:
                        current_auction_player = player_queue[0]
                        pos_str = format_positions(current_auction_player)
                        
                        print(f"\n{Colors.HEADER}--- Round {round_num} ---{Colors.ENDC}")
                        print(f"Player up for Auction: {Colors.BOLD}{current_auction_player.name}{Colors.ENDC} ({pos_str})")
                        print(f"Stats: PTS: {current_auction_player.pts:.1f} | AST: {current_auction_player.ast:.1f} | REB: {current_auction_player.reb:.1f} | BLK: {current_auction_player.blk:.1f} | STL: {current_auction_player.stl:.1f} | TOV: {current_auction_player.tov:.1f} | TS: {current_auction_player.ts*100:.1f}%")
                        
                        # Print our current team
                        team_names = [f"{p.name} ({format_positions(p)})" for p in our_player.nba_team]
                        print(f"Our Current Roster ({roster_size}/5): {', '.join(team_names) if team_names else 'None'}")
                        print(f"Our Balance: {Colors.BOLD}${our_player.balance}{Colors.ENDC}")
                        
                        # Compute bid
                        missing_position_penalty = status['room_settings']['missing_position_penalty']
                        bid = calculate_bid(
                            bot_name=bot_name,
                            members=members,
                            player_queue=player_queue,
                            missing_position_penalty=missing_position_penalty,
                            balance=our_player.balance,
                            current_team=our_player.nba_team
                        )
                        
                        # Submit bid
                        log_info(f"Submitting bid of {Colors.BOLD}${bid}{Colors.ENDC} for {current_auction_player.name}...")
                        
                        bid_response = client.post(
                            f"/rooms/{room_code}/bid",
                            json={"bid_amount": bid, "round_num": round_num},
                            headers={"X-Player-ID": player_id}
                        )
                        
                        bid_res_data = bid_response.json()
                        if bid_res_data.get("success"):
                            log_success(f"Bid of ${bid} successfully submitted!")
                            bot_last_bid_round = round_num
                        else:
                            log_error(f"Failed to submit bid: {bid_res_data.get('failure_msg')}")
                            # We don't update bot_last_bid_round so we can retry next poll
                    else:
                        # Roster full or out of money
                        if roster_size >= 5:
                            log_info(f"Roster is full ({roster_size}/5). Waiting for other players to finish drafting...")
                        else:
                            log_info(f"Balance is $0. Waiting for free player assignments...")
                        bot_last_bid_round = round_num
            else:
                # round_num == 0: Game is not in progress (either hasn't started or just finished)
                if in_game:
                    # Game just finished!
                    in_game = False
                    bot_last_bid_round = 0
                    last_printed_resolved_pid = None
                    
                    print(f"\n{Colors.OKGREEN}======================================")
                    print("===          GAME OVER!            ===")
                    print(f"======================================{Colors.ENDC}")
                    
                    # Print final standings
                    final_players = [Player(**p) for p in status.get("prev_game_final", [])]
                    # Sort by score descending
                    final_players.sort(key=lambda x: x.score, reverse=True)
                    
                    for idx, p in enumerate(final_players):
                        team_str = ", ".join([f"{player.name} ({format_positions(player)})" for player in p.nba_team])
                        highlight = Colors.BOLD if p.name == bot_name else ""
                        print(f"{idx+1}. {highlight}{p.name}{Colors.ENDC}: Score = {Colors.BOLD}{p.score:.2f}{Colors.ENDC} | Balance = ${p.balance} | Roster = [{team_str}]")
                    print(f"{Colors.OKGREEN}======================================\n{Colors.ENDC}")
                    
                if not waiting_message_printed:
                    log_info("Waiting for game to start...")
                    waiting_message_printed = True
                    
        except httpx.RequestError as exc:
            log_warn(f"Connection error to API: {exc}. Retrying in 2 seconds...")
        except Exception as e:
            log_error(f"An unexpected error occurred in the loop: {e}")
            import traceback
            traceback.print_exc()
            
        time.sleep(2)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="NBA Draft Game Smart Bot")
    parser.add_argument("room_code", help="The room code to join")
    parser.add_argument("--api-url", default="http://localhost:8000", help="The base URL of the game API")
    parser.add_argument("--name", default="SmartBot", help="The base name for the bot")
    
    args = parser.parse_args()
    
    print(f"{Colors.HEADER}======================================")
    print("===    NBA DRAFT GAME - SMART BOT  ===")
    print(f"======================================{Colors.ENDC}")
    print(f"Room Code: {args.room_code}")
    print(f"API URL: {args.api_url}")
    print(f"Bot Name: {args.name}")
    print("======================================\n")
    
    play_bot(args.room_code, args.api_url, args.name)
