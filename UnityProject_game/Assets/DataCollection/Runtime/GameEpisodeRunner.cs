using System.Collections;
using System.Collections.Generic;
using UnityEngine;
#if UNITY_EDITOR
using UnityEditor;
#endif

namespace TennisDataCollection.Runtime
{
    // Play-mode coroutine that runs N tennis rallies using the real Bot.cs /
    // ShotManager.cs hit logic. Each episode:
    //   1. Reset ball velocity + position above BotA with a small randomized offset
    //   2. Give the ball an initial serve velocity aimed at BotB
    //   3. Start recording
    //   4. Wait until 3 hits are detected (via BotHitRelay) or timeout
    //   5. Settle period
    //   6. Save CSV, move to next episode
    public class GameEpisodeRunner : MonoBehaviour
    {
        [Header("Scene refs")]
        public Transform ball;
        public Rigidbody ballBody;
        public TrajectoryRecorder recorder;
        public PaperTennisBot botAAgent;
        public PaperTennisBot botBAgent;

        [Header("Dataset")]
        public int episodesToRun = 10;
        public string split = "test";
        public string outDir = "TennisDataset_game";
        public int seed = 1337;
        public int episodeIdOffset = 0;

        [Header("Episode shape")]
        public int strokesPerEpisode = 3;
        public float episodeTimeoutSec = 12f;
        public float finalCatchTimeoutSec = 2.5f;
        public int maxValidationAttemptsMultiplier = 10;

        [Header("Bot homes")]
        public Vector3 botAHome = new Vector3(0f, 1.62f, 11f);
        public Vector3 botBHome = new Vector3(0f, 1.62f, -11f);

        [Header("Opening shot placement")]
        public Vector3 openingBallOffset = new Vector3(0f, 0.95f, -0.45f);
        public Vector2 openingJitterX = new Vector2(-1.5f, 1.5f);

        [Header("Camera randomization")]
        public bool randomizeCamera = true;
        public float camHeightMin = 3f;
        public float camHeightMax = 13f;
        public float camDistMin = 20f;
        public float camDistMax = 45f;
        public float camLateralMin = -0.5f;
        public float camLateralMax = 0.5f;
        public float camFxMin = 1000f;
        public float camFxMax = 3000f;
        public int camOutW = 1280;
        public int camOutH = 720;

        [Header("Terminal frame")]
        public float groundY = 1.295f;
        public float terminalGroundSlack = 0.10f;
        public float terminalStillSpeed = 1.0f;
        public float terminalOutOfBoundsX = 8.0f;
        public float terminalOutOfBoundsZ = 22.0f;

        [Tooltip("If true, alternate which bot serves each episode for both-side coverage.")]
        public bool alternateServer = true;

        int strokesRecorded;
        readonly List<int> hitFrames = new List<int>();
        System.Random rng;
        bool awaitingTerminalCatch;
        bool terminalBoundaryObserved;
        float finalStrokeStartTime;
        PaperTennisBot terminalReceiver;
        bool episodeInvalid;
        string invalidReason;

        void Start()
        {
            rng = new System.Random(seed);
            Application.runInBackground = true;
            Time.fixedDeltaTime = 1f / 50f;
            StartCoroutine(RunAll());
        }

        public void OnBotHit(PaperTennisBot hitter, Vector3 landing)
        {
            if (episodeInvalid) return;
            if (recorder == null || !recorder.IsRecording) return;
            if (strokesRecorded >= strokesPerEpisode) return;

            recorder.FlagHitOnPreviousFrame();
            hitFrames.Add(Mathf.Max(0, recorder.FrameCount - 1));
            strokesRecorded += 1;

            if (hitter != null) hitter.ReturnHome();

            bool allowNextHit = strokesRecorded < strokesPerEpisode;
            var receiver = hitter != null ? hitter.Opponent : null;
            if (receiver != null)
            {
                receiver.PrepareIncomingShot(landing, allowNextHit);
            }

            awaitingTerminalCatch = !allowNextHit;
            terminalReceiver = awaitingTerminalCatch ? receiver : null;
            finalStrokeStartTime = awaitingTerminalCatch ? Time.time : 0f;
        }

        public void InvalidateCurrentEpisode(string reason)
        {
            if (episodeInvalid) return;

            episodeInvalid = true;
            invalidReason = string.IsNullOrWhiteSpace(reason) ? "invalid_episode" : reason;
            awaitingTerminalCatch = false;
            terminalBoundaryObserved = false;
            terminalReceiver = null;

            if (botAAgent != null) botAAgent.DisableHitting();
            if (botBAgent != null) botBAgent.DisableHitting();
        }

        IEnumerator RunAll()
        {
            int valid = 0;
            int attempted = 0;
            int budget = episodesToRun * maxValidationAttemptsMultiplier;
            while (valid < episodesToRun && attempted < budget)
            {
                attempted++;
                bool ok = false;
                yield return RunOne(episodeIdOffset + valid, success => ok = success);
                if (ok) valid++;
            }
            Debug.Log($"[GameEpisodeRunner] DONE valid={valid} attempted={attempted} out={outDir}");
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
            strokesRecorded = 0;
            hitFrames.Clear();
            awaitingTerminalCatch = false;
            terminalBoundaryObserved = false;
            terminalReceiver = null;
            episodeInvalid = false;
            invalidReason = null;

            // Alternate which bot serves each episode so the rally direction
            // varies and both court halves see ball traffic.
            bool aServes = !alternateServer || (episodeId % 2 == 0);
            PaperTennisBot server = aServes ? botAAgent : botBAgent;
            PaperTennisBot receiver = aServes ? botBAgent : botAAgent;
            ResetSceneForEpisode(aServes);

            // Stroke 1 is the opening shot. Recording starts immediately after
            // launch so the sequence begins with the first paper-style parabola,
            // not with a pre-hit settling period.
            Vector3 openingLanding = Vector3.zero;
            if (server == null || !server.LaunchOpeningShot(out openingLanding))
            {
                recorder.BeginRecording();
                yield return new WaitForFixedUpdate();
                recorder.EndRecording(markLastAsEoT: true);
                Debug.Log($"[ep attempt] invalid before opening shot: {invalidReason ?? "opening_shot_failed"} — skipped");
                onDone?.Invoke(false);
                yield break;
            }
            strokesRecorded = 1;
            receiver.PrepareIncomingShot(openingLanding, allowHit: strokesPerEpisode > 1);

            recorder.BeginRecording();
            float t0 = Time.time;

            while ((Time.time - t0) < episodeTimeoutSec)
            {
                yield return new WaitForFixedUpdate();
                if (episodeInvalid) break;
                if (!awaitingTerminalCatch) continue;
                if (TerminalBoundaryReached())
                {
                    terminalBoundaryObserved = true;
                    hitFrames.Add(Mathf.Max(0, recorder.FrameCount - 1));
                    break;
                }
                if ((Time.time - finalStrokeStartTime) > finalCatchTimeoutSec)
                {
                    break;
                }
            }

            recorder.EndRecording(markLastAsEoT: true);

            bool success = !episodeInvalid && strokesRecorded >= strokesPerEpisode && terminalBoundaryObserved;
            if (success)
            {
                var stats = recorder.WriteEpisode(outDir, episodeId, split, hitFrames);
                Debug.Log($"[ep {episodeId}] frames={stats.numFrames} strokes={stats.numStrokes} maxH={stats.maxHeight:F2}m");
            }
            else
            {
                string reason = episodeInvalid ? $" invalid={invalidReason};" : string.Empty;
                Debug.Log($"[ep attempt]{reason} strokes={strokesRecorded}/{strokesPerEpisode} terminal={terminalBoundaryObserved} — skipped");
            }
            onDone?.Invoke(success);
        }

        bool TerminalBoundaryReached()
        {
            if (terminalReceiver != null && terminalReceiver.IsBallStrikeable())
            {
                return true;
            }

            if (ball == null || ballBody == null) return true;

            Vector3 pos = ball.position;
            Vector3 vel = ballBody.linearVelocity;
            if (Mathf.Abs(pos.x) > terminalOutOfBoundsX || Mathf.Abs(pos.z) > terminalOutOfBoundsZ)
            {
                return true;
            }

            return pos.y <= (groundY + terminalGroundSlack) && vel.sqrMagnitude <= (terminalStillSpeed * terminalStillSpeed);
        }

        void ResetSceneForEpisode(bool aServes)
        {
            Vector3 posA = botAHome + new Vector3(RandR(openingJitterX.x, openingJitterX.y), 0f, 0f);
            Vector3 posB = botBHome + new Vector3(RandR(openingJitterX.x, openingJitterX.y), 0f, 0f);

            if (botAAgent != null) botAAgent.ResetForEpisode(posA);
            if (botBAgent != null) botBAgent.ResetForEpisode(posB);

            PaperTennisBot server = aServes ? botAAgent : botBAgent;
            PaperTennisBot receiver = aServes ? botBAgent : botAAgent;
            if (receiver != null) receiver.ReturnHome();

            ballBody.linearVelocity = Vector3.zero;
            ballBody.angularVelocity = Vector3.zero;

            float sideSign = Mathf.Sign(server != null ? server.HomePosition.z : 0f);
            if (sideSign == 0f) sideSign = aServes ? 1f : -1f;
            Vector3 ballOffset = new Vector3(
                RandR(openingJitterX.x, openingJitterX.y),
                openingBallOffset.y,
                -sideSign * Mathf.Abs(openingBallOffset.z));
            ball.position = (server != null ? server.HomePosition : Vector3.zero) + ballOffset;

            if (randomizeCamera)
                RandomizeCamera();
        }

        void RandomizeCamera()
        {
            var cam = Camera.main;
            if (cam == null) return;

            float height = RandR(camHeightMin, camHeightMax);
            float dist = RandR(camDistMin, camDistMax);
            float lateral = RandR(camLateralMin, camLateralMax);

            cam.transform.position = new Vector3(lateral, height + groundY, -dist);
            cam.transform.LookAt(new Vector3(0f, groundY, 0f), Vector3.up);

            float fx = RandR(camFxMin, camFxMax);
            float fovY = 2f * Mathf.Atan((camOutH * 0.5f) / fx) * Mathf.Rad2Deg;
            cam.fieldOfView = fovY;
        }

        float RandR(float a, float b) => (float)(a + rng.NextDouble() * (b - a));
    }
}
