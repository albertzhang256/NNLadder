import arcade
import pytest
from main import (
    SCREEN_WIDTH,
    SCREEN_HEIGHT,
    PLAYER_SPEED,
    FIRE_COOLDOWN,
    MAX_CHARGE_TIME,
    CHARGE_COOLDOWN,
    Bullet,
    Player,
    Enemy,
    GameView,
)


@pytest.fixture
def game_view():
    """Fixture that initializes GameView in headless/offscreen state for unit testing."""
    view = GameView()
    view.setup()
    return view


def test_initial_setup(game_view):
    """Verify initial game state after setup()."""
    assert len(game_view.player_list) == 1
    assert len(game_view.enemy_list) == 6
    assert len(game_view.bullet_list) == 0
    assert game_view.score == 0
    assert game_view.level == 1
    assert game_view.player.center_x == SCREEN_WIDTH // 2


def test_player_movement_and_bounds():
    """Verify player moves properly and clamps within screen boundaries."""
    player = Player()

    # Test moving left past the boundary
    player.center_x = 35
    player.change_x = -PLAYER_SPEED * 5
    player.on_update(1 / 60)
    assert player.center_x == 30  # Clamped at left minimum

    # Test moving right past the boundary
    player.center_x = SCREEN_WIDTH - 35
    player.change_x = PLAYER_SPEED * 5
    player.on_update(1 / 60)
    assert player.center_x == SCREEN_WIDTH - 30  # Clamped at right maximum


def test_key_set_simultaneous_input(game_view):
    """Verify holding both LEFT and RIGHT cancels movement, and releasing one resumes."""
    game_view.on_key_press(arcade.key.LEFT, 0)
    game_view.on_key_press(arcade.key.RIGHT, 0)
    game_view.on_update(1 / 60)
    assert game_view.player.change_x == 0

    # Release LEFT while still holding RIGHT
    game_view.on_key_release(arcade.key.LEFT, 0)
    game_view.on_update(1 / 60)
    assert game_view.player.change_x == PLAYER_SPEED


def test_shooting_and_cooldown(game_view):
    """Verify shooting spawns a bullet, sets cooldown, and respects delta_time."""
    game_view.shoot(play_audio=False)
    assert len(game_view.bullet_list) == 1
    assert game_view.player.shoot_timer == pytest.approx(FIRE_COOLDOWN)

    # Advance half the cooldown time; player should not fire again yet
    game_view.on_key_press(arcade.key.SPACE, 0)
    game_view.on_update(FIRE_COOLDOWN / 2)
    assert len(game_view.bullet_list) == 1

    # Advance past the remaining cooldown time; a second bullet should spawn
    game_view.on_update(FIRE_COOLDOWN)
    assert len(game_view.bullet_list) == 2


def test_charged_shot_growth_and_release(game_view):
    """Verify holding charge button grows width instead of length, and enters 0.5s cooldown."""
    # Press charge key (C)
    game_view.on_key_press(arcade.key.C, 0)
    assert game_view.player.is_charging is True
    assert game_view.player.charge_time == 0.0

    # Initial length scale is 0.8
    assert game_view.charge_sprite.scale[0] == pytest.approx(0.8)

    # Half charge progression: length stays at 0.8, width grows
    game_view.on_update(MAX_CHARGE_TIME / 2)
    assert game_view.player.charge_time == pytest.approx(MAX_CHARGE_TIME / 2)
    assert game_view.charge_sprite.scale[0] == pytest.approx(0.8)  # Length stays constant
    assert game_view.charge_sprite.scale[1] > 2.0  # Width grows

    # Full charge progression: length stays at 0.8, width expands to max
    game_view.on_update(MAX_CHARGE_TIME)
    assert game_view.player.charge_time == pytest.approx(MAX_CHARGE_TIME)
    assert game_view.charge_sprite.scale[0] == pytest.approx(0.8)  # Length stays constant
    assert game_view.charge_sprite.scale[1] == pytest.approx(8.0)  # Width reaches 8.0

    # Release charge key to fire
    game_view.on_key_release(arcade.key.C, 0)
    assert len(game_view.bullet_list) == 1

    charged_bullet = game_view.bullet_list[0]
    assert charged_bullet.scale[0] == pytest.approx(0.8)  # Length remains constant
    assert charged_bullet.scale[1] == pytest.approx(8.0)  # High width
    assert charged_bullet.damage == 50
    assert charged_bullet.is_charged is True
    assert game_view.player.is_charging is False
    assert game_view.player.charge_time == 0.0
    assert game_view.player.charge_cooldown_timer == pytest.approx(CHARGE_COOLDOWN)

    # Cannot charge immediately during 0.5s cooldown
    game_view.on_key_press(arcade.key.C, 0)
    assert game_view.player.is_charging is False

    # After cooldown expires, can charge again
    game_view.on_update(CHARGE_COOLDOWN)
    assert game_view.player.charge_cooldown_timer == 0.0
    game_view.on_key_press(arcade.key.C, 0)
    assert game_view.player.is_charging is True


def test_charged_shot_penetration_and_damage(game_view):
    """Verify charged bullet pierces through multiple enemies and deals high damage."""
    # Setup two aligned enemies: 1 Scout (3 HP) and 1 Cruiser (10 HP)
    scout = Enemy(SCREEN_WIDTH // 2, 400, enemy_type="scout", hp=3)
    cruiser = Enemy(SCREEN_WIDTH // 2, 415, enemy_type="cruiser", hp=10)
    extra_enemy = Enemy(100, 550, enemy_type="scout", hp=3)

    game_view.enemy_list.clear()
    game_view.enemy_list.append(scout)
    game_view.enemy_list.append(cruiser)
    game_view.enemy_list.append(extra_enemy)

    # Fire fully charged shot (damage = 50) directly aligned with wide horizontal scale
    bullet = game_view.shoot(play_audio=False, damage=50, scale=(0.8, 8.0), is_charged=True)
    bullet.center_x = scout.center_x
    bullet.center_y = scout.center_y

    game_view.on_update(1 / 60)

    # Scout and cruiser both pierced and destroyed
    assert scout not in game_view.enemy_list
    assert cruiser not in game_view.enemy_list
    assert extra_enemy in game_view.enemy_list
    assert game_view.score == 40  # 10 (scout) + 30 (cruiser)
    # The charged bullet continues without being destroyed
    assert len(game_view.bullet_list) == 1
    assert bullet in game_view.bullet_list


def test_enemy_types_and_higher_health(game_view):
    """Verify Scout, Cruiser, and Heavy enemy types have increased hit points."""
    scout = Enemy(100, 300, enemy_type="scout")
    cruiser = Enemy(200, 300, enemy_type="cruiser")
    heavy = Enemy(300, 300, enemy_type="heavy")

    assert scout.hp == 3
    assert cruiser.hp == 10
    assert heavy.hp == 25

    # Test cruiser taking 1 hit drops from 10 to 9 HP and survives
    game_view.enemy_list.clear()
    game_view.enemy_list.append(cruiser)

    b1 = game_view.shoot(play_audio=False, damage=1, is_charged=False)
    b1.center_x = cruiser.center_x
    b1.center_y = cruiser.center_y
    game_view.on_update(1 / 60)

    assert cruiser.hp == 9
    assert cruiser in game_view.enemy_list
    assert len(game_view.bullet_list) == 0  # Standard bullet absorbed


def test_levels_increasing_difficulty(game_view):
    """Verify defeating waves triggers next levels with more enemies and higher HP."""
    # Level 1 has 6 enemies with 3 HP each
    assert game_view.level == 1
    assert len(game_view.enemy_list) == 6
    assert all(e.hp == 3 for e in game_view.enemy_list)

    # Defeat all enemies in level 1
    game_view.enemy_list.clear()
    game_view.on_update(1 / 60)

    # Advances to level 2 with 8 enemies including Cruisers (HP 10)
    assert game_view.level == 2
    assert len(game_view.enemy_list) == 8
    cruisers = [e for e in game_view.enemy_list if e.enemy_type == "cruiser"]
    assert len(cruisers) == 3
    assert all(c.hp == 10 for c in cruisers)

    # Defeat all enemies in level 2
    game_view.enemy_list.clear()
    game_view.on_update(1 / 60)

    # Advances to level 3 with 11 enemies including Heavy enemies (HP 25)
    assert game_view.level == 3
    assert len(game_view.enemy_list) == 11
    heavies = [e for e in game_view.enemy_list if e.enemy_type == "heavy"]
    assert len(heavies) == 2
    assert all(h.hp == 25 for h in heavies)


def test_offscreen_bullet_cleanup(game_view):
    """Verify bullets that travel above SCREEN_HEIGHT are garbage-collected."""
    game_view.shoot(play_audio=False)
    bullet = game_view.bullet_list[0]
    bullet.bottom = SCREEN_HEIGHT + 10

    game_view.on_update(1 / 60)
    assert len(game_view.bullet_list) == 0


def test_game_reset(game_view):
    """Verify pressing R resets level, score, and spawns Level 1 enemies."""
    game_view.score = 500
    game_view.level = 4
    game_view.on_key_press(arcade.key.R, 0)

    assert game_view.score == 0
    assert game_view.level == 1
    assert len(game_view.enemy_list) == 6


def test_charged_shot_render():
    """Verify on_draw() executes cleanly while charging and during cooldown in an active window."""
    window = arcade.Window(SCREEN_WIDTH, SCREEN_HEIGHT, visible=False)
    view = GameView()
    view.setup()
    window.show_view(view)

    view.on_key_press(arcade.key.C, 0)
    view.on_update(0.5)
    view.on_draw()

    # Release fires charged shot and enters 0.5s cooldown
    view.on_key_release(arcade.key.C, 0)
    view.on_update(1 / 60)
    # on_draw tests rendering the cooldown indicator
    view.on_draw()

    window.close()
