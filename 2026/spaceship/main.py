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
SCREEN_TITLE = "Extensible 2D Shooter MVP"
PLAYER_SPEED = 50
BULLET_SPEED = 10
FIRE_COOLDOWN = 0.01  # Seconds between shots


class Player(arcade.Sprite):
    """Custom Player class keeps player state and logic self-contained."""
    def __init__(self):
        super().__init__(":resources:images/space_shooter/playerShip1_orange.png", scale=0.5)
        self.center_x = SCREEN_WIDTH // 2
        self.center_y = 50
        self.shoot_timer = 0.0

    def on_update(self, delta_time: float = 1 / 60):
        # Move and clamp to screen bounds
        self.center_x += self.change_x
        self.center_x = max(30, min(SCREEN_WIDTH - 30, self.center_x))
        if self.shoot_timer > 0:
            self.shoot_timer = max(0.0, self.shoot_timer - delta_time)

    update = on_update


class Enemy(arcade.Sprite):
    """Custom Enemy class makes it easy to add HP or AI movement later."""
    def __init__(self, x: float, y: float):
        super().__init__(":resources:images/space_shooter/playerShip1_green.png", scale=0.5)
        self.center_x = x
        self.center_y = y
        self.angle = 180
        self.hp = 1


class GameView(arcade.View):
    """Using a View makes adding Start/Game Over screens effortless later."""
    def __init__(self, window: arcade.Window | None = None):
        try:
            super().__init__(window=window)
        except RuntimeError:
            self.window = None

        if self.window:
            arcade.set_background_color(arcade.color.DARK_MIDNIGHT_BLUE)
        self.laser_sound = arcade.load_sound(":resources:sounds/hurt5.wav")

        self.player_list = None
        self.bullet_list = None
        self.enemy_list = None
        self.player = None
        self.keys_pressed = set()
        self.score = 0
        self.score_text = None

    def setup(self):
        """Initialize or reset the game state."""
        self.player_list = arcade.SpriteList()
        self.bullet_list = arcade.SpriteList()
        self.enemy_list = arcade.SpriteList()
        self.keys_pressed.clear()

        self.player = Player()
        self.player_list.append(self.player)

        for x in range(100, 750, 120):
            self.enemy_list.append(Enemy(x, SCREEN_HEIGHT - 80))

        self.score = 0
        if self.window:
            self.score_text = arcade.Text(
                f"Score: {self.score}", 10, 10, arcade.color.WHITE, 16
            )

    def on_show_view(self):
        if self.score_text is None and self.window:
            self.score_text = arcade.Text(
                f"Score: {self.score}", 10, 10, arcade.color.WHITE, 16
            )

    def on_draw(self):
        self.clear()
        self.player_list.draw()
        self.bullet_list.draw()
        self.enemy_list.draw()
        if self.score_text:
            self.score_text.draw()

    def on_update(self, delta_time: float):
        # Smooth input handling from key set
        left = arcade.key.LEFT in self.keys_pressed or arcade.key.A in self.keys_pressed
        right = arcade.key.RIGHT in self.keys_pressed or arcade.key.D in self.keys_pressed
        self.player.change_x = (right - left) * PLAYER_SPEED

        # Update sprites
        self.player_list.on_update(delta_time)
        self.bullet_list.update()
        self.enemy_list.on_update(delta_time)

        if arcade.key.SPACE in self.keys_pressed and self.player.shoot_timer <= 0:
            self.shoot()

        # Collisions and cleanup
        for bullet in list(self.bullet_list):
            hit_list = arcade.check_for_collision_with_list(bullet, self.enemy_list)
            if hit_list:
                bullet.remove_from_sprite_lists()
                for enemy in hit_list:
                    enemy.hp -= 1
                    if enemy.hp <= 0:
                        enemy.remove_from_sprite_lists()
                        self.score += 1
                        if self.score_text:
                            self.score_text.text = f"Score: {self.score}"
            elif bullet.bottom > SCREEN_HEIGHT:
                bullet.remove_from_sprite_lists()

    def shoot(self, play_audio: bool = True):
        self.player.shoot_timer = FIRE_COOLDOWN
        bullet = arcade.Sprite(":resources:images/space_shooter/laserBlue01.png", scale=0.8)
        bullet.angle = 90
        bullet.center_x = self.player.center_x
        bullet.bottom = self.player.top
        bullet.change_y = BULLET_SPEED
        self.bullet_list.append(bullet)
        if play_audio and self.laser_sound:
            arcade.play_sound(self.laser_sound)

    def on_key_press(self, key, modifiers):
        self.keys_pressed.add(key)
        if key == arcade.key.R:
            self.setup()

    def on_key_release(self, key, modifiers):
        self.keys_pressed.discard(key)


def main():
    window = arcade.Window(SCREEN_WIDTH, SCREEN_HEIGHT, SCREEN_TITLE)
    game_view = GameView()
    game_view.setup()
    window.show_view(game_view)
    arcade.run()


if __name__ == "__main__":
    main()

