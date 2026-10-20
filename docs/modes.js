/* Forms of speaking, their model texts, and what each form actually asks of a voice.
   These forms want opposite things, so one scorecard would be useless: a eulogy delivered at
   Henry V's intensity is a disaster, and the reverse is worse. */

/* Bands are [low, high] of the target range.
   wpm   words per minute over the whole reading, pauses included
   artic syllables per second of actual phonation, pauses excluded
   pitch p5-to-p95 spread of fundamental frequency, in semitones
   dyn   loud-vs-quiet contrast, p90 minus p10 of half-second peak levels, in dB
   ppm   pauses of 0.3 s or more per minute
   plen  their mean length in seconds
   mono  share of 3-second windows sitting inside a 2-semitone band */
export const MODES = {
  oratory: {
    label: "Oratory",
    tag: "Rouse a room that may not want to move",
    cite: "William Shakespeare, <em>Henry V</em>, Act IV scene iii, 1599. Public domain.",
    focus: ["pitch", "dyn"],
    bands: { wpm: [108, 145], artic: [3.9, 5.0], pitch: [12, 18], dyn: [8, 15], ppm: [12, 20], plen: [0.9, 2.0], mono: [0, 15] },
    text: `This day is call'd the feast of Crispian. He that outlives this day, and comes safe home, will stand a tip-toe when this day is nam'd, and rouse him at the name of Crispian.
He that shall live this day, and see old age, will yearly on the vigil feast his neighbours, and say "To-morrow is Saint Crispian." Then will he strip his sleeve and show his scars, and say "These wounds I had on Crispin's day."
Old men forget; yet all shall be forgot, but he'll remember, with advantages, what feats he did that day. Then shall our names, familiar in his mouth as household words, be in their flowing cups freshly rememb'red.
This story shall the good man teach his son; and Crispin Crispian shall ne'er go by, from this day to the ending of the world, but we in it shall be remembered. We few, we happy few, we band of brothers.
For he to-day that sheds his blood with me shall be my brother; be he ne'er so vile, this day shall gentle his condition.`
  },
  ceremonial: {
    label: "Ceremonial",
    tag: "Mark an occasion with gravity, not volume",
    cite: "Abraham Lincoln, Address at Gettysburg, 19 November 1863. Public domain.",
    focus: ["plen", "wpm"],
    bands: { wpm: [95, 125], artic: [3.5, 4.6], pitch: [7, 12], dyn: [5, 10], ppm: [14, 22], plen: [1.0, 2.2], mono: [0, 25] },
    text: `Four score and seven years ago our fathers brought forth on this continent a new nation, conceived in liberty, and dedicated to the proposition that all men are created equal.
Now we are engaged in a great civil war, testing whether that nation, or any nation so conceived and so dedicated, can long endure. We are met on a great battlefield of that war. We have come to dedicate a portion of that field as a final resting place for those who here gave their lives that that nation might live. It is altogether fitting and proper that we should do this.
But in a larger sense, we cannot dedicate, we cannot consecrate, we cannot hallow this ground. The brave men, living and dead, who struggled here have consecrated it far above our poor power to add or detract. The world will little note, nor long remember, what we say here, but it can never forget what they did here.
It is for us the living, rather, to be dedicated here to the unfinished work which they who fought here have thus far so nobly advanced. It is rather for us to be here dedicated to the great task remaining before us, that from these honoured dead we take increased devotion to that cause for which they gave the last full measure of devotion, that we here highly resolve that these dead shall not have died in vain, that this nation shall have a new birth of freedom, and that government of the people, by the people, for the people, shall not perish from the earth.`
  },
  executive: {
    label: "Executive briefing",
    tag: "Land a decision with busy people",
    cite: "Written for this exercise in the product-briefing register. No famous briefing is out of copyright, so this stands in for one.",
    focus: ["artic", "plen"],
    bands: { wpm: [140, 175], artic: [4.7, 5.7], pitch: [9, 14], dyn: [5, 11], ppm: [12, 18], plen: [0.6, 1.3], mono: [0, 20] },
    text: `Three numbers matter this quarter, and only three. Renewal sits at sixty-eight percent. Support cost per account has climbed for five straight months. New revenue is flat.
Those three are the same problem wearing different clothes. Customers who struggle in the first thirty days renew at half the rate of customers who do not, and every one of them calls support twice as often. We have been treating the symptom at the end and ignoring the cause at the start.
So here is what we are asking for. Move four engineers onto onboarding for one quarter. Not a redesign. A single guided path that gets an account to its first useful result inside a week.
If it works, renewal returns to eighty percent, support cost falls with it, and the sales team stops selling against our own reputation. If it does not work, we will know by March, and we will have spent one quarter finding out instead of three.
That is the decision in front of you. One quarter, four people, one number to watch.`
  },
  toast: {
    label: "Wedding toast",
    tag: "Warm, brief, and about them",
    cite: "Written for this exercise. Toasts have no public-domain canon, so this is a model in the right register.",
    focus: ["pitch", "plen"],
    bands: { wpm: [132, 170], artic: [4.4, 5.5], pitch: [10, 16], dyn: [5, 12], ppm: [10, 18], plen: [0.6, 1.5], mono: [0, 18] },
    text: `I have known Daniel since we were nineteen and sharing a flat with a broken radiator. He used to wear a coat indoors and insist it was fine. That is the man. Uncomplaining to a fault, and quietly certain that things will work out.
I met Priya four years later, in a kitchen, at the end of a party nobody wanted to leave. She asked me two questions about myself and then remembered both answers a year afterwards. I understood the appeal immediately.
Here is the thing I did not expect. Daniel got funnier. Not louder. Funnier. He started finishing the thought instead of letting it go. I have watched him become more himself, not less, which is the only test of a marriage I know how to apply.
So I am not going to tell you they are perfect for each other. I am going to tell you something better. They are kind to each other on ordinary days, when nobody is watching and there is nothing to celebrate.
Please raise your glass. To Daniel and Priya.`
  },
  eulogy: {
    label: "Eulogy",
    tag: "Hold steady while saying something true",
    cite: "Thucydides, the Funeral Oration of Pericles, from the Crawley translation of 1874. Public domain.",
    focus: ["dyn", "plen"],
    bands: { wpm: [88, 120], artic: [3.4, 4.5], pitch: [6, 11], dyn: [4, 9], ppm: [14, 22], plen: [1.0, 2.5], mono: [0, 30] },
    text: `So and such they were, these men, worthy of their city. We who remain behind may pray to be spared their bitterness, but must resolve to keep the same daring spirit against the foe.
It is not the oratory of others that can measure the benefit these men have conferred. It is their deeds. For heroes have the whole earth for their tomb; and in lands far from their own, where the column with its epitaph declares it, there is enshrined in every breast a record unwritten, with no tablet to preserve it except that of the heart.
These take as your model. Judge happiness to be the fruit of freedom, and freedom the fruit of valour, and never decline the dangers of war.
For it is not the wretched, who have no hope of better days, that have most cause to be careless of their lives. It is those for whom a change of fortune can still mean a fall, and to whom the difference would be greatest if anything went amiss.
Comfort, therefore, not condolence, is what I have to offer.`
  },
  persuasion: {
    label: "Persuasion",
    tag: "Press a case against resistance",
    cite: "Cicero, the First Oration against Catiline, 63 BC, from the Yonge translation of 1856. Public domain.",
    focus: ["pitch", "dyn"],
    bands: { wpm: [128, 170], artic: [4.5, 5.6], pitch: [11, 17], dyn: [8, 14], ppm: [12, 20], plen: [0.7, 1.6], mono: [0, 16] },
    text: `When, O Catiline, do you mean to cease abusing our patience? How long is that madness of yours still to mock us? When is there to be an end to that unbridled audacity of yours, swaggering about as it does now?
Do not the nightly guards placed on the Palatine Hill move you? Do not the watches posted throughout the city? Does not the alarm of the people, and the union of all good men? Does not the precaution taken of assembling the senate in this most defensible place? Do not the looks and countenances of this venerable body here present have any effect upon you?
Do you not feel that your plans are detected? Do you not see that your conspiracy is already arrested and rendered powerless by the knowledge which every one here possesses of it?
Shame on the age, and on its principles. The senate is aware of these things. The consul sees them. And yet this man lives. Lives? He comes even into the senate. He takes part in the public deliberations. He marks out with his eyes, for slaughter, every single one of us.`
  }
};

export const METRICS = [
  { key: "wpm", label: "Pace", unit: " wpm", axis: [70, 210], dec: 0,
    low: "Slower than this form wants. The gap is usually hunting for the next word, not deliberate weight.",
    high: "Faster than this form wants. Put the time back as silence after full stops, not inside clauses." },
  { key: "artic", label: "Articulation rate", unit: " syll/s", axis: [2.5, 7], dec: 1,
    low: "Syllables are stretched. Right for grief, costly anywhere else.",
    high: "Syllables are compressed. Long words are where this shows first." },
  { key: "pitch", label: "Pitch range", unit: " st", axis: [2, 22], dec: 1,
    low: "Too narrow for this form. Go up on a new idea, down to finish one.",
    high: "Wider than the form needs. Reads as performed rather than meant." },
  { key: "dyn", label: "Volume contrast", unit: " dB", axis: [1, 18], dec: 1,
    low: "One level throughout. Drop to near-quiet for the line that matters, then come back.",
    high: "Swinging hard. Keep the floor up so the quiet passages still carry." },
  { key: "ppm", label: "Pauses per minute", unit: "/min", axis: [2, 30], dec: 1,
    low: "Sentences are running together. Stop at the full stops.",
    high: "Breaking up more often than the form needs. Rehearse phrases as whole units." },
  { key: "plen", label: "Mean pause", unit: " s", axis: [0.2, 3], dec: 2,
    low: "Pauses are clipped. A landed pause is about a second; shorter reads as a breath.",
    high: "Pauses are long. Deliberate is good, searching is not." },
  { key: "mono", label: "Monotone stretches", unit: "%", axis: [0, 60], dec: 0,
    low: "", high: "Long passages on one note. Mark two or three lines per page to lift or drop." }
];

/* ------------------------------------------------------------------ text */

/* Words the vowel-group rule gets wrong and that are common enough in speech to matter. */
const IRREGULAR = { business: 2, businesses: 3, evening: 2, interesting: 3, different: 3,
  difference: 3, family: 3, comfortable: 4, vegetable: 3, chocolate: 3, average: 3, several: 3,
  general: 3, federal: 3, camera: 3, memory: 3, everything: 3, everyone: 3, people: 2, little: 2,
  science: 2, quiet: 2, being: 2, doing: 2, going: 2, poem: 2, create: 2 };

/** Syllables in one word. Exact on a 50-word reference set drawn from real transcripts, with no
 *  net bias; the bare vowel-group rule scored 72% and read "management" as four. */
export function syllables(word) {
  const w = word.toLowerCase().replace(/[^a-z]/g, "");
  if (!w) return 0;
  if (IRREGULAR[w] !== undefined) return IRREGULAR[w];
  if (w.length <= 3) return 1;
  // Written without regex lookbehind on purpose: Safari only shipped it in 16.4, and a
  // SyntaxError here would stop the whole app loading on an older iPhone.
  let t = w.replace(/e(ments?|ness|less|ful|ly)$/, "$1");   // silent e before a consonant suffix
  if (/[^laeiouy]es$/.test(t)) t = t.replace(/[^laeiouy]es$/, "");
  // "-ed" is its own syllable after t or d (ded-i-cat-ed, want-ed) but silent elsewhere (walked)
  else if (/ed$/.test(t) && !/[td]ed$/.test(t)) t = t.replace(/ed$/, "");
  else if (/[^laeiouy]e$/.test(t)) t = t.replace(/[^laeiouy]e$/, "");
  t = t.replace(/^y/, "");
  const m = t.match(/[aeiouy]{1,2}/g);
  let n = Math.max(1, m ? m.length : 1);
  // a final "ia"/"ea" is two syllables (i-de-a), and "-ian" is too after most consonants
  // (cris-pi-an) but not after c, t or s (o-cean, chris-tian). Both need another vowel group
  // earlier in the word, so "plea" and "sea" stay at one.
  if (n >= 2 && /(ia|ea)$/.test(w)) n += 1;
  else if (n >= 2 && /[^ctsaeiou]ian$/.test(w)) n += 1;
  return n;
}

export const words = t => t.match(/[A-Za-z'’]+/g) || [];
/** Split into sentences, keeping any closing quote with the sentence it ends.
 *  Splitting on whitespace after a terminator (the obvious approach) silently fails on
 *  `he said "...Crispian." Then` because the character before the space is the quote, not the
 *  full stop, so two sentences merge into one. It also avoids lookbehind, which Safari before
 *  16.4 cannot parse. */
export function sentencesOf(t) {
  const out = [];
  const re = /[^.!?]*[.!?]+["'\u201d\u2019)\]]*/g;
  let m, last = 0;
  while ((m = re.exec(t)) !== null) {
    if (!m[0]) break;
    if (m[0].trim()) out.push(m[0].trim());
    last = re.lastIndex;
  }
  const tail = t.slice(last).trim();
  if (tail) out.push(tail);
  return out;
}

/** Words, sentences and syllables of a passage: everything pace needs without transcription. */
export function scriptStats(text) {
  const ws = words(text);
  return {
    words: ws.length,
    sentences: sentencesOf(text.replace(/\n+/g, " ")).length,
    syllables: ws.reduce((a, w) => a + syllables(w), 0),
  };
}
