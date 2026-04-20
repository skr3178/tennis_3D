using System.Collections;
using System.Collections.Generic;
using UnityEngine;
#if UNITY_EDITOR
using UnityEditor;
#endif

namespace TennisDataCollection.Runtime
{
    // Paper Sec 3.2 style: "simulate a bouncing ball by applying a series of
    // impulse forces". No bot AI. Same Unity scene, ball, ground, camera, and
    // PhysX physics material as the tennis tutorial — only the chase logic is
    // skipped. Each episode:
    //   1. Reset ball at one baseline with a randomized x.
    //   2. Apply launch velocity toward the opposite side (random speed/angle).
    //   3. Wait until the ball touches the ground (one bounce).
    //   4. Wait until it reaches near-apex of the post-bounce arc.
    //   5. Apply a return velocity at the ball's current position toward the
    //      original side. Repeat for `strokesPerEpisode` strokes.
    //   6. Settle, save trajectory.
    public class ImpulseEpisodeRunner : MonoBehaviour
    {
        [Header("Scene refs")]
        public Transform ball;
        public Rigidbody ballBody;
        public TrajectoryRecorder recorder;

        [Header("Dataset")]
        public int episodesToRun = 10;
        public string split = "test";
        public string outDir = "TennisDataset_impulse";
        public int seed = 1337;

        [Header("Episode shape")]
        public int strokesPerEpisode = 4;
        public float settleSeconds = 1.5f;
        public float episodeTimeoutSec = 12f;

        [Header("Court geometry")]
        public float baselineZ = 10.5f;            // |z| of the launch position
        public Vector2 launchXJitter = new Vector2(-3f, 3f);
        public float launchY = 1.2f;               // ball height when struck

        [Header("Stroke params")]
        // Tuned so the resulting parabola lands inside the opponent's half
        // (|z| ~5..10m from net) rather than flying off the back of the court.
        public Vector2 strokeSpeed = new Vector2(13f, 18f);    // m/s
        public Vector2 strokeAngleDeg = new Vector2(12f, 22f); // upward from horizontal
        public Vector2 targetXRange = new Vector2(-3.5f, 3.5f);
        public Vector2 targetZRange = new Vector2(4f, 9f);     // |z| of intended landing

        [Header("Bounce detection")]
        public float groundContactY = 0.15f;       // ball radius ~0.067, so this is near-contact

        readonly List<int> hitFrames = new List<int>();
        System.Random rng;

        void Start()
        {
            rng = new System.Random(seed);
            Application.runInBackground = true;
            Time.fixedDeltaTime = 1f / 50f;
            StartCoroutine(RunAll());
        }

        IEnumerator RunAll()
        {
            int valid = 0;
            int attempted = 0;
            int budget = episodesToRun * 4;
            while (valid < episodesToRun && attempted < budget)
            {
                attempted++;
                bool ok = false;
                yield return RunOne(valid, success => ok = success);
                if (ok) valid++;
            }
            Debug.Log($"[ImpulseEpisodeRunner] DONE valid={valid} attempted={attempted} out={outDir}");
            yield return null;

#if UNITY_EDITOR
            EditorApplication.isPlaying = false;
            EditorApplication.Exit(valid >= episodesToRun ? 0 : 1);
#else
            Application.Quit(valid >= episodesToRun ? 0 : 1);
#endif
        }

        IEnumerator RunOne(int episodeId, System.Action<bool> onDone)
        {
            hitFrames.Clear();

            // First serve: from one baseline, toward the other.
            float dirSign = (episodeId % 2 == 0) ? -1f : +1f;   // alternate which side serves
            float startZ = -dirSign * baselineZ;                 // opposite of where we send it
            Vector3 startPos = new Vector3(RandR(launchXJitter.x, launchXJitter.y), launchY, startZ);
            ball.position = startPos;
            ballBody.linearVelocity = Vector3.zero;
            ballBody.angularVelocity = Vector3.zero;
            yield return new WaitForFixedUpdate();

            recorder.BeginRecording();

            // Apply first stroke immediately at frame 0.
            ApplyStroke(startPos, dirSign);
            hitFrames.Add(Mathf.Max(0, recorder.FrameCount - 1));

            float t0 = Time.time;

            for (int s = 1; s < strokesPerEpisode; s++)
            {
                // --- wait for bounce: vy went negative, then positive while near ground
                bool wasFalling = false;
                bool bounced = false;
                float minY = float.PositiveInfinity;
                Vector3 lastVAtMinY = Vector3.zero;
                Vector3 lastP = Vector3.zero;
                while (Time.time - t0 < episodeTimeoutSec)
                {
                    yield return new WaitForFixedUpdate();
                    Vector3 v = ballBody.linearVelocity;
                    Vector3 p = ball.position;
                    lastP = p;
                    if (p.y < minY) { minY = p.y; lastVAtMinY = v; }
                    if (v.y < 0f) wasFalling = true;
                    if (wasFalling && v.y > 0.5f && p.y < groundContactY * 4f)
                    { bounced = true; break; }
                }
                if (!bounced)
                {
                    Debug.Log($"[diag ep {episodeId} stroke {s}] no bounce. minY={minY:F3} vAtMinY={lastVAtMinY} lastP={lastP}");
                    break;
                }

                // --- wait for near-apex (vy still slightly positive)
                bool atApex = false;
                while (Time.time - t0 < episodeTimeoutSec)
                {
                    yield return new WaitForFixedUpdate();
                    if (ballBody.linearVelocity.y < 0.3f) { atApex = true; break; }
                }
                if (!atApex) break;

                dirSign = -dirSign;
                Vector3 hitPos = ball.position;
                ApplyStroke(hitPos, dirSign);
                recorder.FlagHitOnPreviousFrame();
                hitFrames.Add(Mathf.Max(0, recorder.FrameCount - 1));
            }

            // Let the final shot land + a brief settle so the trajectory
            // shows the closing bounce rather than ending mid-flight.
            float settleStart = Time.time;
            while (Time.time - settleStart < settleSeconds)
                yield return new WaitForFixedUpdate();

            recorder.EndRecording(markLastAsEoT: true);

            bool success = hitFrames.Count >= strokesPerEpisode;
            if (success)
            {
                var stats = recorder.WriteEpisode(outDir, episodeId, split, hitFrames);
                Debug.Log($"[impulse ep {episodeId}] frames={stats.numFrames} strokes={stats.numStrokes} maxH={stats.maxHeight:F2}m");
            }
            else
            {
                Debug.Log($"[impulse ep attempt] strokes={hitFrames.Count}/{strokesPerEpisode} — skipped");
            }
            onDone?.Invoke(success);
        }

        void ApplyStroke(Vector3 from, float dirSignZ)
        {
            float speed = RandR(strokeSpeed.x, strokeSpeed.y);
            float angleRad = RandR(strokeAngleDeg.x, strokeAngleDeg.y) * Mathf.Deg2Rad;
            float tx = RandR(targetXRange.x, targetXRange.y);
            float tz = dirSignZ * RandR(targetZRange.x, targetZRange.y);
            Vector3 horiz = new Vector3(tx - from.x, 0f, tz - from.z);
            if (horiz.sqrMagnitude < 0.0001f) horiz = new Vector3(0f, 0f, dirSignZ);
            horiz.Normalize();
            Vector3 vel = horiz * (speed * Mathf.Cos(angleRad)) + Vector3.up * (speed * Mathf.Sin(angleRad));
            ballBody.linearVelocity = vel;
            ballBody.angularVelocity = Vector3.zero;
        }

        float RandR(float a, float b) => (float)(a + rng.NextDouble() * (b - a));
    }
}
