using System.IO;
using UnityEditor;
using UnityEditor.SceneManagement;
using UnityEngine;
using TennisDataCollection.Runtime;

namespace TennisDataCollection.EditorTools
{
    // Opens the tennis tutorial's edited.unity, but DISABLES the bots and the
    // player keyboard script. Adds the ImpulseEpisodeRunner so the rally is
    // driven by impulses applied to the ball directly (paper Sec 3.2 method).
    // Same court, same ball, same PhysX material — just no chase AI.
    public static class ImpulseSceneBuilder
    {
        public const string SourceScene = "Assets/Scenes/edited.unity";
        public const string OutputScene = "Assets/DataCollection/Scenes/Collection_impulse.unity";

        // Make this big enough that no reasonable impulse sends the ball off
        // the edge — bigger ground = guaranteed bounce.
        const float ExtendedGroundZ = 200f;
        const float ExtendedGroundX = 200f;

        [MenuItem("Tools/Tennis Data/Build Impulse Scene")]
        public static void Build()
        {
            Directory.CreateDirectory("Assets/DataCollection/Scenes");
            var scene = EditorSceneManager.OpenScene(SourceScene, OpenSceneMode.Single);

            var ball = GameObject.Find("ball");
            var cam  = Camera.main;
            if (ball == null || cam == null)
            {
                Debug.LogError($"[ImpulseSceneBuilder] missing ball or Main Camera: ball={ball} cam={cam}");
                return;
            }

            // Larger flat ground so bounces aren't constrained to the small tutorial floor.
            var extGround = GameObject.CreatePrimitive(PrimitiveType.Cube);
            extGround.name = "ExtendedGround";
            extGround.transform.position = new Vector3(0f, -0.06f, 0f);
            extGround.transform.localScale = new Vector3(ExtendedGroundX, 0.1f, ExtendedGroundZ);
            var bouncePM = AssetDatabase.LoadAssetAtPath<PhysicsMaterial>("Assets/bounce 1.physicMaterial");
            if (bouncePM != null) extGround.GetComponent<Collider>().sharedMaterial = bouncePM;
            var rExt = extGround.GetComponent<Renderer>();
            if (rExt != null) rExt.enabled = false;

            // Ball: ensure rigidbody, recorder, and the camera reference.
            if (ball.GetComponent<Rigidbody>() == null)
            {
                var rb = ball.AddComponent<Rigidbody>();
                rb.mass = 0.058f;
                rb.collisionDetectionMode = CollisionDetectionMode.Continuous;
                rb.interpolation = RigidbodyInterpolation.Interpolate;
                rb.useGravity = true;
            }
            var recorder = ball.GetComponent<TrajectoryRecorder>();
            if (recorder == null) recorder = ball.AddComponent<TrajectoryRecorder>();
            recorder.SetCamera(cam);
            recorder.SetOutputResolution(1280, 720);

            // Disable the bot AI, ShotManager, the player keyboard script, AND
            // the tutorial Ball.cs (which zeros velocity on collision with any
            // "Wall"-tagged object). We want clean ballistic + bounce physics.
            DisableComponentByName("Bot");
            DisableComponentByName("ShotManager");
            DisableComponentByName("Player");
            DisableComponentByName("Ball");

            // Disable any "Wall"-tagged colliders too, just in case. The
            // tutorial scene has back walls that catch the ball mid-flight.
            foreach (var col in Object.FindObjectsByType<Collider>(FindObjectsSortMode.None))
            {
                if (col != null && col.gameObject.CompareTag("Wall")) col.enabled = false;
            }

            // Also park the bot GameObjects far off-court so their colliders
            // can't catch the ball mid-air.
            var bot = GameObject.Find("bot");
            if (bot != null) bot.transform.position = new Vector3(0f, -50f, 0f);
            var player = GameObject.Find("player");
            if (player != null) player.transform.position = new Vector3(0f, -50f, 0f);

            // Episode runner.
            var mgrGo = new GameObject("ImpulseRunner");
            var runner = mgrGo.AddComponent<ImpulseEpisodeRunner>();
            runner.ball = ball.transform;
            runner.ballBody = ball.GetComponent<Rigidbody>();
            runner.recorder = recorder;

            EditorSceneManager.SaveScene(scene, OutputScene);
            Debug.Log($"[ImpulseSceneBuilder] saved -> {OutputScene}");
        }

        static void DisableComponentByName(string typeName)
        {
            foreach (var mb in Object.FindObjectsByType<MonoBehaviour>(FindObjectsSortMode.None))
            {
                if (mb == null) continue;
                if (mb.GetType().Name == typeName) mb.enabled = false;
            }
        }
    }
}
