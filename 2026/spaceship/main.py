import arcade

# Forward-compatibility for Arcade versions
if not hasattr(arcade.SpriteList, "on_update"):
    def _sprite_list_on_update(self, delta_time: float = 1 / 60):
        for sprite in self:
            if hasattr(sprite, "on_update"):
                sprite.on_update(delta_time)
            elif hasattr(sprite, "update"):
                sprite.update(delta_time)
    arcade.SpriteList.on_update = _sprite_list_on_update

# --- Constants ---
SCREEN_WIDTH = 800
SCREEN_HEIGHT = 600
SCREEN_TITLE = "2D Space Shooter"
PLAYER_SPEED = 10
BULLET_SPEED = 20
FIRE_COOLDOWN = 0.1  # Rapid blaster cooldown
MAX_CHARGE_TIME = 3.0  # Increased charge time in seconds to reach full charge
CHARGE_COOLDOWN = 0.5  # Seconds of cooldown required between charged shots
CHARGE_KEYS = {arcade.key.C, arcade.key.LSHIFT, arcade.key.RSHIFT, arcade.key.X}


class Bullet(arcade.Sprite):
    """Bullet supporting variable damage, scale, and penetration for charged shots."""
    def __init__(self, x: float, y: float, damage: int = 1, scale: float = 0.8, is_charged: bool = False):
        super().__init__(":resources:images/space_shooter/laserBlue01.png", scale=scale)
        self.angle = 90
        self.center_x = x
        self.bottom = y
        self.change_y = BULLET_SPEED
        self.damage = damage
        self.is_charged = is_charged
        self.hit_enemies = set()


class Player(arcade.Sprite):
    """Player entity managing bounds clamping, shoot cooldown, and charge state."""
    def __init__(self):
        super().__init__(":resources:images/space_shooter/playerShip1_orange.png", scale=0.5)
        self.center_x = SCREEN_WIDTH // 2
        self.center_y = 50
        self.shoot_timer = 0.0
        self.charge_time = 0.0
        self.charge_cooldown_timer = 0.0
        self.is_charging = False

    def on_update(self, delta_time: float = 1 / 60):
        # Move and clamp to screen bounds
        self.center_x += self.change_x
        self.center_x = max(30, min(SCREEN_WIDTH - 30, self.center_x))
        if self.shoot_timer > 0:
            self.shoot_timer = max(0.0, self.shoot_timer - delta_time)

        if self.charge_cooldown_timer > 0:
            self.charge_cooldown_timer = max(0.0, self.charge_cooldown_timer - delta_time)

        if self.is_charging:
            self.charge_time = min(MAX_CHARGE_TIME, self.charge_time + delta_time)

    update = on_update


class Enemy(arcade.Sprite):
    """Custom Enemy class supporting multiple enemy types, HPs, and movement patterns."""
    def __init__(
        self,
        x: float,
        y: float,
        enemy_type: str = "scout",
        hp: int | None = None,
        speed_x: float = 0.0,
        speed_y: float = 0.0,
    ):
        self.enemy_type = enemy_type
        if enemy_type == "scout":
            super().__init__(":resources:images/space_shooter/playerShip1_green.png", scale=0.5)
            self.max_hp = hp if hp is not None else 3
            self.points = 10
            self.angle = 180
        elif enemy_type == "cruiser":
            super().__init__(":resources:images/space_shooter/playerShip1_blue.png", scale=0.55)
            self.max_hp = hp if hp is not None else 10
            self.points = 30
            self.angle = 180
        elif enemy_type == "heavy":
            super().__init__(":resources:images/space_shooter/meteorGrey_big1.png", scale=0.65)
            self.max_hp = hp if hp is not None else 25
            self.points = 60
            self.angle = 0
        else:
            super().__init__(":resources:images/space_shooter/playerShip1_green.png", scale=0.5)
            self.max_hp = hp if hp is not None else 3
            self.points = 10
            self.angle = 180

        self.center_x = x
        self.center_y = y
        self.hp = self.max_hp
        self.change_x = speed_x
        self.change_y = speed_y

    def on_update(self, delta_time: float = 1 / 60):
        self.center_x += self.change_x
        self.center_y += self.change_y

        # Bounce horizontally off screen walls
        if self.left <= 15:
            self.left = 15
            self.change_x = abs(self.change_x)
        elif self.right >= SCREEN_WIDTH - 15:
            self.right = SCREEN_WIDTH - 15
            self.change_x = -abs(self.change_x)

        # Wrap if reaching bottom
        if self.top <= 0:
            self.bottom = SCREEN_HEIGHT + 20

    update = on_update


class GameView(arcade.View):
    """Main view handling player input, shooting/charging, enemy waves, and HUD."""
    def __init__(self, window: arcade.Window | None = None):
        try:
            super().__init__(window=window)
        except RuntimeError:
            self.window = None

        if self.window:
            arcade.set_background_color(arcade.color.DARK_MIDNIGHT_BLUE)

        self.laser_sound = arcade.load_sound(":resources:sounds/hurt5.wav")
        self.charged_sound = arcade.load_sound(":resources:sounds/laser1.wav")
        self.level_up_sound = arcade.load_sound(":resources:sounds/upgrade1.wav")

        self.player_list = None
        self.bullet_list = None
        self.enemy_list = None
        self.charge_list = None
        self.player = None
        self.charge_sprite = None
        self.keys_pressed = set()
        self.score = 0
        self.level = 1
        self.level_banner_timer = 0.0

        self.score_text = None
        self.controls_text = None
        self.level_banner_text = None

    def setup(self):
        """Initialize or reset the game state."""
        self.player_list = arcade.SpriteList()
        self.bullet_list = arcade.SpriteList()
        self.enemy_list = arcade.SpriteList()
        self.charge_list = arcade.SpriteList()
        self.keys_pressed.clear()

        self.player = Player()
        self.player_list.append(self.player)

        # Sprite used to visualize the growing charge at the ship nose
        self.charge_sprite = arcade.Sprite(":resources:images/space_shooter/laserBlue01.png", scale=(0.8, 0.8))
        self.charge_sprite.angle = 90
        self.charge_list.append(self.charge_sprite)

        self.score = 0
        self.level = 1
        self.level_banner_timer = 0.0
        self.spawn_level(self.level)

        if self.window:
            self.init_text()

    def init_text(self):
        self.score_text = arcade.Text(
            f"Score: {self.score}   Level: {self.level}", 15, 15, arcade.color.WHITE, 16
        )
        self.level_banner_text = arcade.Text(
            f"LEVEL {self.level}",
            SCREEN_WIDTH // 2,
            SCREEN_HEIGHT // 2,
            arcade.color.GOLD,
            32,
            anchor_x="center",
            anchor_y="center",
        )
        self.controls_text = arcade.Text(
            "A/D: Move | SPACE: Blaster | Hold C or SHIFT: Charge Shot | R: Reset",
            SCREEN_WIDTH // 2,
            SCREEN_HEIGHT - 20,
            arcade.color.LIGHT_GRAY,
            11,
            anchor_x="center",
        )

    def on_show_view(self):
        if self.score_text is None and self.window:
            self.init_text()

    def spawn_level(self, level: int):
        """Spawn enemies for the given level with increasing difficulty."""
        self.enemy_list.clear()

        if level == 1:
            # Level 1: 6 Scouts across top row (3 HP each)
            for x in range(100, 750, 120):
                self.enemy_list.append(Enemy(x, SCREEN_HEIGHT - 80, enemy_type="scout", hp=3, speed_x=0.0, speed_y=0.0))
        elif level == 2:
            # Level 2: 8 enemies - 5 Scouts (3 HP) + 3 Cruisers (10 HP)
            for x in range(100, 750, 150):
                self.enemy_list.append(Enemy(x, SCREEN_HEIGHT - 80, enemy_type="scout", hp=3, speed_x=1.5, speed_y=-0.1))
            for x in range(200, 650, 180):
                self.enemy_list.append(Enemy(x, SCREEN_HEIGHT - 160, enemy_type="cruiser", hp=10, speed_x=-1.2, speed_y=-0.1))
        elif level == 3:
            # Level 3: 11 enemies - 5 Scouts (4 HP), 4 Cruisers (12 HP), 2 Heavies (25 HP)
            for x in range(80, 750, 150):
                self.enemy_list.append(Enemy(x, SCREEN_HEIGHT - 70, enemy_type="scout", hp=4, speed_x=2.0, speed_y=-0.15))
            for x in range(120, 720, 160):
                self.enemy_list.append(Enemy(x, SCREEN_HEIGHT - 150, enemy_type="cruiser", hp=12, speed_x=-1.8, speed_y=-0.15))
            for x in range(250, 600, 250):
                self.enemy_list.append(Enemy(x, SCREEN_HEIGHT - 230, enemy_type="heavy", hp=25, speed_x=1.0, speed_y=-0.2))
        else:
            # Level 4+: Procedural difficulty scaling
            num_scouts = min(8, 4 + level)
            num_cruisers = min(6, 2 + level // 2)
            num_heavies = min(4, 1 + (level - 2) // 2)
            base_speed = 1.0 + 0.2 * (level - 1)

            step_s = (SCREEN_WIDTH - 160) / max(1, num_scouts - 1)
            for i in range(num_scouts):
                x = 80 + i * step_s
                self.enemy_list.append(Enemy(x, SCREEN_HEIGHT - 70, enemy_type="scout", hp=3 + level, speed_x=base_speed, speed_y=-0.2))

            step_c = (SCREEN_WIDTH - 200) / max(1, num_cruisers - 1)
            for i in range(num_cruisers):
                x = 100 + i * step_c
                self.enemy_list.append(Enemy(x, SCREEN_HEIGHT - 150, enemy_type="cruiser", hp=10 + level * 2, speed_x=-base_speed * 0.9, speed_y=-0.2))

            step_h = (SCREEN_WIDTH - 300) / max(1, num_heavies - 1) if num_heavies > 1 else 0
            for i in range(num_heavies):
                x = 150 + i * step_h if num_heavies > 1 else SCREEN_WIDTH // 2
                self.enemy_list.append(Enemy(x, SCREEN_HEIGHT - 230, enemy_type="heavy", hp=25 + level * 4, speed_x=base_speed * 0.6, speed_y=-0.25))

    def on_draw(self):
        self.clear()
        self.player_list.draw()
        self.bullet_list.draw()
        self.enemy_list.draw()

        # Render health bars for tougher enemies (HP > 1)
        for enemy in self.enemy_list:
            if enemy.max_hp > 1:
                bar_w = 40
                bar_h = 4
                left = enemy.center_x - bar_w / 2
                bottom = enemy.top + 6
                # Background
                arcade.draw_lbwh_rectangle_filled(left, bottom, bar_w, bar_h, arcade.color.DARK_RED)
                # Foreground
                hp_ratio = max(0.0, enemy.hp / enemy.max_hp)
                color = arcade.color.GREEN if hp_ratio > 0.4 else arcade.color.YELLOW
                arcade.draw_lbwh_rectangle_filled(left, bottom, bar_w * hp_ratio, bar_h, color)

        # Draw charging orb and gauge while charging
        if self.player.is_charging and self.player.charge_time > 0.05 and self.charge_list:
            self.charge_list.draw()
            charge_ratio = min(1.0, self.player.charge_time / MAX_CHARGE_TIME)
            gauge_w = 40
            gauge_left = self.player.center_x - gauge_w / 2
            gauge_bottom = self.player.bottom - 10
            arcade.draw_lbwh_rectangle_filled(gauge_left, gauge_bottom, gauge_w, 4, arcade.color.DARK_BLUE)
            charge_color = arcade.color.CYAN if charge_ratio < 1.0 else arcade.color.GOLD
            arcade.draw_lbwh_rectangle_filled(gauge_left, gauge_bottom, gauge_w * charge_ratio, 4, charge_color)
        elif self.player.charge_cooldown_timer > 0:
            # Draw cooldown indicator while on charge cooldown
            gauge_w = 40
            gauge_left = self.player.center_x - gauge_w / 2
            gauge_bottom = self.player.bottom - 10
            cd_ratio = self.player.charge_cooldown_timer / CHARGE_COOLDOWN
            arcade.draw_lbwh_rectangle_filled(gauge_left, gauge_bottom, gauge_w, 3, arcade.color.DARK_RED)
            arcade.draw_lbwh_rectangle_filled(gauge_left, gauge_bottom, gauge_w * cd_ratio, 3, arcade.color.ORANGE)

        if self.score_text:
            self.score_text.draw()
        if self.controls_text:
            self.controls_text.draw()
        if self.level_banner_timer > 0 and self.level_banner_text:
            self.level_banner_text.draw()

    def on_update(self, delta_time: float):
        # Smooth input handling from key set
        left = arcade.key.LEFT in self.keys_pressed or arcade.key.A in self.keys_pressed
        right = arcade.key.RIGHT in self.keys_pressed or arcade.key.D in self.keys_pressed
        self.player.change_x = (right - left) * PLAYER_SPEED

        # Update sprites
        self.player_list.on_update(delta_time)
        self.bullet_list.update()
        self.enemy_list.on_update(delta_time)

        # Update charge visual indicator position and growing width (not length)
        if self.player.is_charging:
            charge_ratio = min(1.0, self.player.charge_time / MAX_CHARGE_TIME)
            scale_length = 0.8
            scale_width = 0.8 + 7.2 * charge_ratio
            self.charge_sprite.scale = (scale_length, scale_width)
            self.charge_sprite.center_x = self.player.center_x
            self.charge_sprite.bottom = self.player.top + 2
        elif any(k in self.keys_pressed for k in CHARGE_KEYS):
            # If holding charge key and cooldown expires, start charging
            if self.player.charge_cooldown_timer <= 0:
                self.player.is_charging = True
                self.player.charge_time = 0.0
                self.charge_sprite.scale = (0.8, 0.8)

        if self.level_banner_timer > 0:
            self.level_banner_timer = max(0.0, self.level_banner_timer - delta_time)

        # Standard blaster autofire
        if arcade.key.SPACE in self.keys_pressed and self.player.shoot_timer <= 0 and not self.player.is_charging:
            self.shoot()

        # Collisions, damage, and piercing
        for bullet in list(self.bullet_list):
            if bullet.bottom > SCREEN_HEIGHT:
                bullet.remove_from_sprite_lists()
                continue

            hit_list = arcade.check_for_collision_with_list(bullet, self.enemy_list)
            if hit_list:
                for enemy in hit_list:
                    if bullet.is_charged:
                        # Piercing charged shot: damages each enemy in flight path once and penetrates
                        if enemy not in bullet.hit_enemies:
                            bullet.hit_enemies.add(enemy)
                            enemy.hp -= bullet.damage
                            if enemy.hp <= 0:
                                enemy.remove_from_sprite_lists()
                                self.score += enemy.points
                                if self.score_text:
                                    self.score_text.text = f"Score: {self.score}   Level: {self.level}"
                    else:
                        # Standard bullet: single hit and consumed
                        enemy.hp -= bullet.damage
                        if enemy.hp <= 0:
                            enemy.remove_from_sprite_lists()
                            self.score += enemy.points
                            if self.score_text:
                                self.score_text.text = f"Score: {self.score}   Level: {self.level}"
                        bullet.remove_from_sprite_lists()
                        break

        # Check for wave completion and advance level
        if len(self.enemy_list) == 0:
            self.level += 1
            self.spawn_level(self.level)
            self.level_banner_timer = 2.0
            if self.level_banner_text:
                self.level_banner_text.text = f"LEVEL {self.level}"
            if self.score_text:
                self.score_text.text = f"Score: {self.score}   Level: {self.level}"
            if self.level_up_sound:
                arcade.play_sound(self.level_up_sound)

    def shoot(self, play_audio: bool = True, damage: int = 1, scale: float = 0.8, is_charged: bool = False):
        self.player.shoot_timer = FIRE_COOLDOWN
        bullet = Bullet(
            x=self.player.center_x,
            y=self.player.top,
            damage=damage,
            scale=scale,
            is_charged=is_charged,
        )
        self.bullet_list.append(bullet)
        if play_audio:
            sound = self.charged_sound if is_charged else self.laser_sound
            if sound:
                arcade.play_sound(sound)
        return bullet

    def fire_charged_shot(self, play_audio: bool = True):
        """Release the charged shot with width and damage proportional to charge duration."""
        charge_ratio = min(1.0, self.player.charge_time / MAX_CHARGE_TIME)
        scale_length = 0.8
        scale_width = 0.8 + 7.2 * charge_ratio
        damage = 5 + int(45 * charge_ratio)  # 5 damage at min charge up to 50 damage at full charge
        self.player.is_charging = False
        self.player.charge_time = 0.0
        self.player.charge_cooldown_timer = CHARGE_COOLDOWN  # 0.5s cooldown
        return self.shoot(play_audio=play_audio, damage=damage, scale=(scale_length, scale_width), is_charged=True)

    def on_key_press(self, key, modifiers):
        self.keys_pressed.add(key)
        if key == arcade.key.R:
            self.setup()
        elif key in CHARGE_KEYS:
            if self.player.charge_cooldown_timer <= 0:
                self.player.is_charging = True
                self.player.charge_time = 0.0
                self.charge_sprite.scale = (0.8, 0.8)

    def on_key_release(self, key, modifiers):
        self.keys_pressed.discard(key)
        if key in CHARGE_KEYS and self.player.is_charging:
            self.fire_charged_shot()


def main():
    window = arcade.Window(SCREEN_WIDTH, SCREEN_HEIGHT, SCREEN_TITLE)
    game_view = GameView()
    game_view.setup()
    window.show_view(game_view)
    arcade.run()


if __name__ == "__main__":
    main()
