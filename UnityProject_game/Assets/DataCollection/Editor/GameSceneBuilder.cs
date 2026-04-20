using System.IO;
using UnityEditor;
using UnityEditor.SceneManagement;
using UnityEngine;
using TennisDataCollection.Runtime;

namespace TennisDataCollection.EditorTools
{
    // Opens the game's real scene (Assets/Scenes/edited.unity), finds the
    // existing ball / bot / Main Camera GameObjects and patches them with our
    // data-collection components. Then duplicates the bot so the rally is
    // bot-vs-bot. Saves as Assets/DataCollection/Scenes/Collection_game.unity
    // so the source scene stays untouched.
    public static class GameSceneBuilder
    {
        public const string SourceScene = "Assets/Scenes/edited.unity";
        public const string OutputScene = "Assets/DataCollection/Scenes/Collection_game.unity";

        // Paper-like homes near the baselines. The paper's receiver can move
        // 5-7m behind the predicted landing spot, so the data spans farther
        // than the tutorial scene's original court.
        static readonly Vector3 BotAPosition = new Vector3(0f, 1.62f,  11.0f);
        static readonly Vector3 Bot2Position = new Vector3(0f, 1.62f, -11.0f);
        const float ExtendedGroundZ = 46f;
        const float ExtendedGroundX = 24f;

        // The tutorial scene's `ground` GameObject sits with its top surface at
        // world y ≈ 1.295 (parent y=2.589 + child y=-1.543 + half scale 0.25).
        // Subtract this from saved coords so the dataset matches paper convention
        // (court at y=0).
        const float GroundOffsetY = 1.295f;

        // Paper broadcast camera spec (Appendix B.2): FOV 28°, position (0, 14, -30)
        // looking at origin, image 1280x720. Camera world Y is shifted up by
        // GroundOffsetY so that AFTER y-offset is applied at write time, the
        // saved camera lands at the paper's spec.
        static readonly Vector3 PaperCamPosition = new Vector3(0f, 14f + GroundOffsetY, -30f);
        static readonly Vector3 PaperCamLookAt   = new Vector3(0f,  0f + GroundOffsetY,   0f);
        const float PaperCamFovDeg = 28f;
        const int PaperCamW = 1280;
        const int PaperCamH = 720;

        [MenuItem("Tools/Tennis Data/Build Collection Scene (from real game)")]
        public static void Build()
        {
            Directory.CreateDirectory("Assets/DataCollection/Scenes");

            var scene = EditorSceneManager.OpenScene(SourceScene, OpenSceneMode.Single);

            var ball = GameObject.Find("ball");
            var botA = GameObject.Find("bot");
            var cam  = Camera.main;
            if (ball == null || botA == null || cam == null)
            {
                Debug.LogError($"[GameSceneBuilder] missing required objects: ball={ball} bot={botA} cam={cam}");
                return;
            }

            // Add a larger hidden bounce surface so the paper-style receiver can
            // stand behind deep landings without falling off the tutorial court.
            var extGround = GameObject.CreatePrimitive(PrimitiveType.Cube);
            extGround.name = "ExtendedGround";
            extGround.transform.position = new Vector3(0f, GroundOffsetY - 0.05f, 0f);
            extGround.transform.localScale = new Vector3(ExtendedGroundX, 0.1f, ExtendedGroundZ);
            var bouncePM = AssetDatabase.LoadAssetAtPath<PhysicsMaterial>("Assets/bounce 1.physicMaterial");
            if (bouncePM != null) extGround.GetComponent<Collider>().sharedMaterial = bouncePM;
            // Disable the mesh renderer so it doesn't visually clash with the
            // existing textured ground — we only need it as a physics surface.
            var r = extGround.GetComponent<Renderer>();
            if (r != null) r.enabled = false;

            // Move botA to the regulation end-position.
            botA.transform.position = BotAPosition;

            // ---------- Ball ----------
            EnsureTagExists("Ball");
            ball.tag = "Ball";
            if (ball.GetComponent<Rigidbody>() == null)
            {
                var ballRb = ball.AddComponent<Rigidbody>();
                ballRb.mass = 0.058f;
                ballRb.collisionDetectionMode = CollisionDetectionMode.Continuous;
                ballRb.interpolation = RigidbodyInterpolation.Interpolate;
                ballRb.useGravity = true;
            }
            var recorder = ball.GetComponent<TrajectoryRecorder>();
            if (recorder == null) recorder = ball.AddComponent<TrajectoryRecorder>();
            recorder.SetCamera(cam);
            recorder.SetGroundOffsetY(GroundOffsetY);
            recorder.SetOutputResolution(PaperCamW, PaperCamH);
            StripComponent<Ball>(ball);

            // Override the tutorial camera with paper's broadcast spec.
            cam.transform.position = PaperCamPosition;
            cam.transform.LookAt(PaperCamLookAt, Vector3.up);
            cam.fieldOfView = PaperCamFovDeg;

            // ---------- Bot B (duplicate of bot A) ----------
            var botB = Object.Instantiate(botA);
            botB.name = "bot2";
            botB.transform.position = Bot2Position;
            botB.transform.rotation = Quaternion.Euler(0f, 180f, 0f) * botA.transform.rotation;

            // ---------- Remove any player scripts (keyboard-driven) ----------
            // `enabled = false` only stops Update/FixedUpdate; OnTriggerEnter
            // still fires and dereferences a null aimTarget (Player.cs:77).
            // Destroy the component outright so its trigger callbacks vanish.
            foreach (var p in Object.FindObjectsByType<Player>(FindObjectsSortMode.None))
            {
                Object.DestroyImmediate(p);
            }

            // The collection scene must not inherit the tutorial bot/wall logic:
            // Bot.cs only tracks x, and Ball.cs teleports on wall contact.
            StripComponent<Bot>(botA);
            StripComponent<Bot>(botB);

            var rb = ball.GetComponent<Rigidbody>();
            var botAAgent = ConfigurePaperBot(botA, ball.transform, rb);
            var botBAgent = ConfigurePaperBot(botB, ball.transform, rb);
            botAAgent.opponent = botBAgent;
            botBAgent.opponent = botAAgent;

            // ---------- Episode runner ----------
            var mgrGo = new GameObject("EpisodeRunner");
            var runner = mgrGo.AddComponent<GameEpisodeRunner>();
            runner.ball = ball.transform;
            runner.ballBody = rb;
            runner.recorder = recorder;
            runner.botAAgent = botAAgent;
            runner.botBAgent = botBAgent;
            runner.botAHome = BotAPosition;
            runner.botBHome = Bot2Position;
            runner.groundY = GroundOffsetY;

            botAAgent.runner = runner;
            botBAgent.runner = runner;
            ConfigureCourtCollision(runner);

            EditorSceneManager.SaveScene(scene, OutputScene);
            Debug.Log($"[GameSceneBuilder] saved -> {OutputScene}   ball={ball.transform.position} " +
                      $"botA={botA.transform.position} botB={botB.transform.position} cam={cam.transform.position}");
        }

        static PaperTennisBot ConfigurePaperBot(GameObject bot, Transform ball, Rigidbody ballBody)
        {
            var agent = bot.GetComponent<PaperTennisBot>();
            if (agent == null) agent = bot.AddComponent<PaperTennisBot>();
            agent.ball = ball;
            agent.ballBody = ballBody;
            agent.groundY = GroundOffsetY;
            agent.netZ = 0f;
            agent.landingXRange = new Vector2(-4.1f, 4.1f);
            agent.landingAbsZRange = new Vector2(0.5f, 11.0f);
            agent.launchAngleDegRange = new Vector2(10f, 20f);
            agent.preferredSpeedRange = new Vector2(14f, 22f);
            agent.launchSampleAttempts = 40;
            agent.ballisticSolveAttempts = 12;
            agent.netHeightAboveGround = 1.07f;
            agent.netClearanceMargin = 0.15f;
            agent.netHalfWidth = 5.8f;
            agent.receiverOffsetRange = new Vector2(5f, 7f);
            agent.receiverMoveSpeed = 40f;
            agent.maxAbsReceiverX = 6.5f;
            agent.maxAbsReceiverZ = 18.5f;
            return agent;
        }

        static void ConfigureCourtCollision(GameEpisodeRunner runner)
        {
            foreach (var col in Object.FindObjectsByType<Collider>(FindObjectsSortMode.None))
            {
                if (!col) continue;

                if (IsNetHierarchy(col.transform))
                {
                    col.enabled = true;
                    var validator = col.GetComponent<NetClearanceValidator>();
                    if (validator == null) validator = col.gameObject.AddComponent<NetClearanceValidator>();
                    validator.runner = runner;
                    var netRenderer = col.GetComponent<Renderer>();
                    if (netRenderer != null) netRenderer.enabled = true;
                    continue;
                }

                if (!col.CompareTag("Wall")) continue;
                col.enabled = false;
                var renderer = col.GetComponent<Renderer>();
                if (renderer != null) renderer.enabled = false;
            }
        }

        static bool IsNetHierarchy(Transform node)
        {
            for (var current = node; current != null; current = current.parent)
            {
                if (current.name.IndexOf("net", System.StringComparison.OrdinalIgnoreCase) >= 0)
                {
                    return true;
                }
            }
            return false;
        }

        static void StripComponent<T>(GameObject go) where T : Component
        {
            if (go == null) return;
            var component = go.GetComponent<T>();
            if (component != null) Object.DestroyImmediate(component);
        }

        // Re-binds a bot's `targets` array to point at hand-picked positions on the
        // opposite side of the net. aimAtSideSign = −1 aims at player half (−z);
        // +1 aims at bot's own original side (+z).
        static void EnsureBotTargetsCorrect(GameObject bot, float aimAtSideSign)
        {
            var botScript = bot.GetComponent<Bot>();
            if (botScript == null) return;

            var parent = new GameObject($"{bot.name}_targets");
            parent.transform.SetParent(bot.transform.parent, worldPositionStays: true);

            // Widen target grid to match paper's per-rally x-span (~5.5 m
            // average, max ~8 m). Singles court half-width is 4.115 m; we go
            // slightly wider to allow occasional out-shots like the paper's
            // x range [-8.4, 8.4] in aggregate.
            float[] xs = { -5.5f, -2.75f, 0f, 2.75f, 5.5f };
            // Push targets out to the opponent's baseline (z=11.89) for a longer
            // bounce travel matching the paper's z extent (~17 m one-side).
            float[] zs = { 3.0f, 6.0f, 9.0f, 10.5f };
            var list = new System.Collections.Generic.List<Transform>();
            foreach (var x in xs)
            {
                foreach (var z in zs)
                {
                    var t = new GameObject($"t_{list.Count}").transform;
                    t.SetParent(parent.transform, false);
                    t.position = new Vector3(x, 0f, aimAtSideSign * z);
                    list.Add(t);
                }
            }
            botScript.targets = list.ToArray();

            // aimTarget is declared in Bot.cs but only used as an editor-side hook;
            // point it at the centre of the opponent court so the inspector is tidy.
            if (botScript.aimTarget == null)
            {
                var aim = new GameObject($"{bot.name}_aimTarget").transform;
                aim.SetParent(bot.transform.parent, false);
                aim.position = new Vector3(0f, 1.37f, aimAtSideSign * 5f);
                botScript.aimTarget = aim;
            }
        }

        static void AddReceiverPositioner(GameObject bot, Transform ball)
        {
            var rp = bot.GetComponent<ReceiverPositioner>();
            if (rp == null) rp = bot.AddComponent<ReceiverPositioner>();
            rp.ball = ball;
            rp.ballBody = ball.GetComponent<Rigidbody>();
            rp.groundY = GroundOffsetY;            // tutorial court surface
            rp.receiverOffsetRange = new Vector2(5f, 7f);  // paper App. B.2
            rp.moveSpeed = 8f;
        }

        static void PatchShotManagers(GameObject bot, float hitForce, float upForce,
                                      float hitForceFlat, float upForceFlat)
        {
            var sm = bot.GetComponent<ShotManager>();
            if (sm == null) return;
            if (sm.topSpin != null) { sm.topSpin.hitForce = hitForce;     sm.topSpin.upForce = upForce; }
            if (sm.flat    != null) { sm.flat.hitForce    = hitForceFlat; sm.flat.upForce    = upForceFlat; }
        }

        static void EnsureTagExists(string tag)
        {
            var asset = AssetDatabase.LoadAllAssetsAtPath("ProjectSettings/TagManager.asset");
            if (asset == null || asset.Length == 0) return;
            var so = new SerializedObject(asset[0]);
            var tagsProp = so.FindProperty("tags");
            for (int i = 0; i < tagsProp.arraySize; i++)
                if (tagsProp.GetArrayElementAtIndex(i).stringValue == tag) return;
            tagsProp.arraySize++;
            tagsProp.GetArrayElementAtIndex(tagsProp.arraySize - 1).stringValue = tag;
            so.ApplyModifiedProperties();
        }
    }
}
