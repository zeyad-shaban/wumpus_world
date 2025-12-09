import pygame
import math


class AgentInspectorWindow:
    """Enhanced agent state visualizer with decision queue and probability maps"""
    
    def __init__(self, agent, grid_width, grid_height):
        """
        Initialize the inspector
        
        Args:
            agent: The AI agent to inspect
            grid_width: Width of the game grid
            grid_height: Height of the game grid
        """
        self.agent = agent
        self.grid_width = grid_width
        self.grid_height = grid_height
        self.font = pygame.font.SysFont(None, 16)
        self.title_font = pygame.font.SysFont(None, 22)
        self.small_font = pygame.font.SysFont(None, 12)
        
        # Risk map mode: 'pit_prob', 'wumpus_prob', 'death_prob'
        self.risk_mode = 'death_prob'
        
        # Track the y position where risk tabs are drawn (for click handling)
        self.risk_tabs_y = 0
        
        # Decision history (last N decisions)
        self.decision_history = []
        self.max_history = 10
        
        # Colors for mini-grid
        self.colors = {
            'unknown': (50, 50, 60),
            'visited': (80, 80, 100),
            'agent': (100, 150, 255),
            'wumpus': (255, 80, 80),
            'pit': (40, 40, 40),
            'gold': (255, 215, 0),
            'exit': (100, 255, 150),
            'breeze': (150, 200, 255),
            'stench': (150, 255, 150),
            'safe': (60, 100, 60),
            'danger': (100, 50, 50),
            'grid_line': (70, 70, 80),
        }
    
    def add_decision(self, action, action_probs, chosen_reason):
        """Add a decision to history"""
        decision = {
            'action': action,
            'probs': action_probs.copy() if action_probs else {},
            'reason': chosen_reason,
            'step': len(self.decision_history) + 1
        }
        self.decision_history.append(decision)
        if len(self.decision_history) > self.max_history:
            self.decision_history.pop(0)
    
    def draw(self, surface):
        """
        Draw the agent inspector panel to the given surface
        
        Args:
            surface: Pygame surface to draw on
        """
        # Background
        surface.fill((30, 30, 40))
        
        # Border line separating from game
        pygame.draw.line(surface, (80, 80, 100), (0, 0), (0, surface.get_height()), 2)
        
        y_offset = 8
        panel_width = surface.get_width()
        
        # Title
        title = self.title_font.render("Agent Inspector", True, (255, 255, 255))
        surface.blit(title, (10, y_offset))
        y_offset += 25
        
        # Get agent state if available
        if hasattr(self.agent, 'env') and self.agent.env:
            state = self.agent.env.get_state()
            percepts = self.agent.env.get_percepts()
            
            # Section 1: Overview Map (always visible)
            y_offset = self._draw_mini_grid(surface, y_offset, state, percepts, compact=True)
            y_offset += 5
            
            # Separator
            pygame.draw.line(surface, (80, 80, 100), (10, y_offset), (panel_width - 10, y_offset), 1)
            y_offset += 8
            
            # Section 2: Risk Heatmap with tabs
            y_offset = self._draw_risk_section(surface, y_offset, state)
            y_offset += 5
            
            # Separator
            pygame.draw.line(surface, (80, 80, 100), (10, y_offset), (panel_width - 10, y_offset), 1)
            y_offset += 8
            
            # Section 3: Text info (decision queue, percepts, status)
            y_offset = self._draw_decision_queue(surface, y_offset)
            y_offset += 5
            
            # Separator
            pygame.draw.line(surface, (80, 80, 100), (10, y_offset), (panel_width - 10, y_offset), 1)
            y_offset += 8
            
            # Current percepts
            y_offset = self._draw_percepts(surface, y_offset, percepts)
            y_offset += 5
            
            # Separator
            pygame.draw.line(surface, (80, 80, 100), (10, y_offset), (panel_width - 10, y_offset), 1)
            y_offset += 8
            
            # Status info
            y_offset = self._draw_status(surface, y_offset, state)
        else:
            text = self.font.render("No environment data", True, (150, 150, 150))
            surface.blit(text, (10, y_offset))
    
    def _draw_risk_section(self, surface, y_offset, state):
        """Draw risk heatmap section with mode tabs"""
        panel_width = surface.get_width()
        
        # Section title
        label = self.font.render("Risk Heatmap", True, (200, 200, 200))
        surface.blit(label, (10, y_offset))
        y_offset += 18
        
        # Draw risk mode tabs
        self.risk_tabs_y = y_offset
        y_offset = self._draw_risk_tabs(surface, y_offset)
        y_offset += 5
        
        # Draw the selected risk heatmap
        y_offset = self._draw_probability_map(surface, y_offset, state)
        
        return y_offset
    
    def _draw_risk_tabs(self, surface, y_offset):
        """Draw risk mode selector tabs"""
        modes = [
            ('pit_prob', 'Pit Risk'),
            ('wumpus_prob', 'Wumpus Risk'),
            ('death_prob', 'Death Risk')
        ]
        
        panel_width = surface.get_width()
        total_buttons = len(modes)
        button_width = 85
        spacing = 5
        total_width = total_buttons * button_width + (total_buttons - 1) * spacing
        x_start = (panel_width - total_width) // 2
        button_height = 18
        
        for i, (mode_key, mode_label) in enumerate(modes):
            x = x_start + i * (button_width + spacing)
            
            # Button background
            is_selected = (self.risk_mode == mode_key)
            bg_color = (60, 100, 140) if is_selected else (50, 50, 60)
            border_color = (100, 150, 200) if is_selected else (70, 70, 80)
            
            button_rect = pygame.Rect(x, y_offset, button_width, button_height)
            pygame.draw.rect(surface, bg_color, button_rect)
            pygame.draw.rect(surface, border_color, button_rect, 1)
            
            # Button text
            text_color = (255, 255, 255) if is_selected else (180, 180, 180)
            text = self.small_font.render(mode_label, True, text_color)
            text_rect = text.get_rect(center=button_rect.center)
            surface.blit(text, text_rect)
        
        return y_offset + button_height + 5
    
    def _draw_probability_map(self, surface, y_offset, state):
        """Draw probability heatmap for selected risk type"""
        panel_width = surface.get_width()
        
        # Calculate cell size (smaller for compact view)
        margin = 15
        available_width = panel_width - 2 * margin
        available_height = 140  # Smaller height for risk map
        
        cell_size = min(available_width // self.grid_width, available_height // self.grid_height)
        cell_size = max(cell_size, 12)
        
        grid_pixel_width = cell_size * self.grid_width
        grid_pixel_height = cell_size * self.grid_height
        
        # Center the grid
        grid_x = (panel_width - grid_pixel_width) // 2
        grid_y = y_offset
        
        agent_pos = state['agent']
        
        # Calculate probabilities
        probs = {}
        for row in range(self.grid_height):
            for col in range(self.grid_width):
                pos = (row, col)
                
                if self.risk_mode == 'pit_prob':
                    prob = self.agent.pit_probability(pos)
                elif self.risk_mode == 'wumpus_prob':
                    prob = self.agent.wumpus_probability(pos)
                else:  # death_prob
                    prob = self.agent.death_probability(pos)
                
                probs[pos] = prob
        
        # Draw cells
        for row in range(self.grid_height):
            for col in range(self.grid_width):
                cell_x = grid_x + col * cell_size
                cell_y = grid_y + row * cell_size
                pos = (row, col)
                
                prob = probs[pos]
                
                # Color based on probability (green = safe, red = dangerous)
                if prob == 0:
                    cell_color = (40, 80, 40)  # Safe green
                else:
                    intensity = min(prob / 1.0, 1.0)
                    r = int(255 * intensity)
                    g = int(200 * (1 - intensity))
                    b = 40
                    cell_color = (r, g, b)
                
                # Draw cell
                cell_rect = pygame.Rect(cell_x, cell_y, cell_size - 1, cell_size - 1)
                pygame.draw.rect(surface, cell_color, cell_rect)
                
                # Draw probability text if cell is large enough (lowered threshold)
                if cell_size >= 15:
                    prob_text = f"{int(prob * 100)}"
                    text = self.small_font.render(prob_text, True, (255, 255, 255))
                    text_rect = text.get_rect(center=(cell_x + cell_size // 2, cell_y + cell_size // 2))
                    surface.blit(text, text_rect)
                
                # Highlight agent position
                if pos == agent_pos:
                    pygame.draw.rect(surface, (100, 150, 255), cell_rect, 2)
                    center_x = cell_x + cell_size // 2
                    center_y = cell_y + cell_size // 2
                    pygame.draw.circle(surface, (100, 150, 255), (center_x, center_y), 3)
        
        # Draw grid border
        border_rect = pygame.Rect(grid_x - 1, grid_y - 1, grid_pixel_width + 2, grid_pixel_height + 2)
        pygame.draw.rect(surface, self.colors['grid_line'], border_rect, 1)
        
        # Draw compact legend
        legend_y = grid_y + grid_pixel_height + 3
        legend_width = 150
        legend_height = 10
        legend_x = (panel_width - legend_width) // 2
        
        # Draw gradient bar
        for i in range(legend_width):
            intensity = i / legend_width
            r = int(255 * intensity)
            g = int(200 * (1 - intensity))
            b = 40
            pygame.draw.line(surface, (r, g, b), (legend_x + i, legend_y), (legend_x + i, legend_y + legend_height))
        
        # Draw legend labels
        label_0 = self.small_font.render("0%", True, (180, 180, 180))
        label_100 = self.small_font.render("100%", True, (180, 180, 180))
        surface.blit(label_0, (legend_x - 18, legend_y))
        surface.blit(label_100, (legend_x + legend_width + 3, legend_y))
        
        return legend_y + legend_height + 8
    
    def _draw_mini_grid(self, surface, y_offset, state, percepts, compact=False):
        """Draw a mini version of the game grid showing agent's view"""
        panel_width = surface.get_width()
        
        # Calculate cell size to fit in panel
        margin = 15
        available_width = panel_width - 2 * margin
        available_height = 140 if compact else 200  # Smaller for compact mode
        
        cell_size = min(available_width // self.grid_width, available_height // self.grid_height)
        cell_size = max(cell_size, 12)
        
        grid_pixel_width = cell_size * self.grid_width
        grid_pixel_height = cell_size * self.grid_height
        
        # Center the grid
        grid_x = (panel_width - grid_pixel_width) // 2
        grid_y = y_offset
        
        # Draw title for mini-grid
        label = self.font.render("Overview Map", True, (200, 200, 200))
        surface.blit(label, (grid_x, grid_y))
        grid_y += 18
        
        # Get known information
        agent_pos = state['agent']
        wumpus_positions = set(state['wumpus'])
        pit_positions = state['pits']
        gold_pos = state['gold']
        exit_pos = state['exit']
        
        visited = getattr(self.agent, 'visited', {agent_pos})
        
        # Draw each cell
        for row in range(self.grid_height):
            for col in range(self.grid_width):
                cell_x = grid_x + col * cell_size
                cell_y = grid_y + row * cell_size
                pos = (row, col)
                
                # Determine cell color
                cell_color = self.colors['unknown']
                if pos in visited:
                    cell_color = self.colors['visited']
                
                # Draw base cell
                cell_rect = pygame.Rect(cell_x, cell_y, cell_size - 1, cell_size - 1)
                pygame.draw.rect(surface, cell_color, cell_rect)
                
                # Draw hazards/items
                inner_margin = 2
                inner_size = cell_size - 2 * inner_margin - 1
                
                # Pit
                if pos in pit_positions:
                    pit_rect = pygame.Rect(cell_x + inner_margin, cell_y + inner_margin, 
                                          inner_size, inner_size)
                    pygame.draw.rect(surface, self.colors['pit'], pit_rect)
                    pygame.draw.line(surface, (100, 100, 100), 
                                   (cell_x + inner_margin, cell_y + inner_margin),
                                   (cell_x + cell_size - inner_margin - 1, cell_y + cell_size - inner_margin - 1), 1)
                    pygame.draw.line(surface, (100, 100, 100),
                                   (cell_x + cell_size - inner_margin - 1, cell_y + inner_margin),
                                   (cell_x + inner_margin, cell_y + cell_size - inner_margin - 1), 1)
                
                # Wumpus
                if pos in wumpus_positions:
                    center_x = cell_x + cell_size // 2
                    center_y = cell_y + cell_size // 2
                    radius = inner_size // 3
                    pygame.draw.circle(surface, self.colors['wumpus'], (center_x, center_y), radius)
                
                # Gold
                if gold_pos and pos == gold_pos:
                    center_x = cell_x + cell_size // 2
                    center_y = cell_y + cell_size // 2
                    radius = inner_size // 4
                    pygame.draw.circle(surface, self.colors['gold'], (center_x, center_y), radius)
                
                # Exit
                if pos == exit_pos:
                    exit_rect = pygame.Rect(cell_x + inner_margin + 2, cell_y + inner_margin + 2,
                                           inner_size - 4, inner_size - 4)
                    pygame.draw.rect(surface, self.colors['exit'], exit_rect, 2)
                
                # Percept indicators
                is_adjacent_to_pit = any(
                    (row + dr, col + dc) in pit_positions 
                    for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]
                )
                if is_adjacent_to_pit and pos not in pit_positions:
                    pygame.draw.polygon(surface, self.colors['breeze'], [
                        (cell_x + 1, cell_y + 1),
                        (cell_x + 6, cell_y + 1),
                        (cell_x + 1, cell_y + 6)
                    ])
                
                is_adjacent_to_wumpus = any(
                    (row + dr, col + dc) in wumpus_positions
                    for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]
                )
                if is_adjacent_to_wumpus and pos not in wumpus_positions:
                    pygame.draw.polygon(surface, self.colors['stench'], [
                        (cell_x + cell_size - 2, cell_y + 1),
                        (cell_x + cell_size - 7, cell_y + 1),
                        (cell_x + cell_size - 2, cell_y + 6)
                    ])
                
                # Agent position
                if pos == agent_pos:
                    center_x = cell_x + cell_size // 2
                    center_y = cell_y + cell_size // 2
                    radius = inner_size // 3
                    pygame.draw.circle(surface, self.colors['agent'], (center_x, center_y), radius)
                    pygame.draw.circle(surface, (255, 255, 255), (center_x, center_y), 2)
        
        # Draw grid border
        border_rect = pygame.Rect(grid_x - 1, grid_y - 1, grid_pixel_width + 2, grid_pixel_height + 2)
        pygame.draw.rect(surface, self.colors['grid_line'], border_rect, 1)
        
        return grid_y + grid_pixel_height + 5
    
    def _draw_decision_queue(self, surface, y_offset):
        """Draw recent decision history"""
        text = self.font.render("Decision History:", True, (220, 220, 220))
        surface.blit(text, (10, y_offset))
        y_offset += 16
        
        if not self.decision_history:
            text = self.small_font.render("No decisions yet", True, (120, 120, 120))
            surface.blit(text, (15, y_offset))
            return y_offset + 14
        
        action_names = ['UP', 'DOWN', 'LEFT', 'RIGHT', 'SHOOT_UP', 'SHOOT_DOWN', 'SHOOT_LEFT', 'SHOOT_RIGHT']
        
        # Show last 5 decisions
        visible_history = self.decision_history[-5:]
        
        for i, decision in enumerate(reversed(visible_history)):
            action = decision['action']
            action_name = action_names[action] if action is not None and 0 <= action < len(action_names) else 'NONE'
            
            # Decision text with fade for older decisions
            alpha = 255 - (i * 40)
            alpha = max(alpha, 100)
            
            text_str = f"#{decision['step']}: {action_name}"
            text = self.small_font.render(text_str, True, (alpha, alpha, alpha))
            surface.blit(text, (15, y_offset))
            y_offset += 12
        
        return y_offset + 5
    
    def _draw_percepts(self, surface, y_offset, percepts):
        """Draw current percepts section"""
        text = self.font.render("Current Percepts:", True, (220, 220, 220))
        surface.blit(text, (10, y_offset))
        y_offset += 16
        
        x_offset = 15
        
        # Breeze
        breeze_color = self.colors['breeze'] if percepts['breeze'] else (80, 80, 80)
        pygame.draw.circle(surface, breeze_color, (x_offset + 5, y_offset + 6), 5)
        text = self.small_font.render("Breeze", True, breeze_color)
        surface.blit(text, (x_offset + 13, y_offset + 1))
        x_offset += 60
        
        # Stench
        stench_color = self.colors['stench'] if percepts['stench'] else (80, 80, 80)
        pygame.draw.circle(surface, stench_color, (x_offset + 5, y_offset + 6), 5)
        text = self.small_font.render("Stench", True, stench_color)
        surface.blit(text, (x_offset + 13, y_offset + 1))
        x_offset += 60
        
        # Glitter
        glitter_color = self.colors['gold'] if percepts['glitter'] else (80, 80, 80)
        pygame.draw.circle(surface, glitter_color, (x_offset + 5, y_offset + 6), 5)
        text = self.small_font.render("Glitter", True, glitter_color)
        surface.blit(text, (x_offset + 13, y_offset + 1))
        
        return y_offset + 18
    
    def _draw_status(self, surface, y_offset, state):
        """Draw agent status information"""
        # Position
        text = self.font.render(f"Position: {state['agent']}", True, (200, 200, 200))
        surface.blit(text, (10, y_offset))
        y_offset += 16
        
        # Has gold
        gold_color = (255, 215, 0) if state['have_gold'] else (120, 120, 120)
        text = self.font.render(f"Has Gold: {state['have_gold']}", True, gold_color)
        surface.blit(text, (10, y_offset))
        y_offset += 16
        
        # Arrows and score
        text = self.font.render(f"Arrows: {state['arrows_remaining']}", True, (200, 200, 200))
        surface.blit(text, (10, y_offset))
        
        score_color = (100, 255, 100) if state['total'] >= 0 else (255, 100, 100)
        text = self.font.render(f"Score: {state['total']}", True, score_color)
        surface.blit(text, (100, y_offset))
        y_offset += 16
        
        # Wumpuses remaining
        text = self.font.render(f"Wumpuses: {len(state['wumpus'])}", True, (255, 120, 120))
        surface.blit(text, (10, y_offset))
        
        return y_offset + 16
    
    def handle_click(self, mouse_pos, panel_offset_x):
        """Handle mouse clicks for risk mode selection"""
        # Adjust mouse position relative to panel
        mx, my = mouse_pos
        mx -= panel_offset_x
        
        # Check if click is in risk tabs area
        if self.risk_tabs_y <= my <= self.risk_tabs_y + 25:
            modes = ['pit_prob', 'wumpus_prob', 'death_prob']
            panel_width = 300  # Approximate panel width
            button_width = 85
            spacing = 5
            total_width = len(modes) * button_width + (len(modes) - 1) * spacing
            x_start = (panel_width - total_width) // 2
            
            for i, mode in enumerate(modes):
                button_x = x_start + i * (button_width + spacing)
                if button_x <= mx <= button_x + button_width:
                    self.risk_mode = mode
                    return True
        
        return False